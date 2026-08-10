"""單次訓練＋評估的通用核心：一組超參數 × 一個切分 × 一顆種子 → 一列指標。

`train.py` 回答的是「這個準確率能不能信」（切分協定），本檔是它的下一步：
**在已經可信的切分上，超參數與架構到底能不能改變結論。** 兩者共用同一個
訓練迴圈——`train.py` 現在也走這裡的 `fit_eval`，用 `BASELINE` 這組設定，
且有 `tests/test_experiment_parity.py` 釘住它與既有 README 數字逐位元組相同。
共用的理由很現實：若搜尋用另一套實作，搜出來的贏家可能贏在實作差異上。

刻意保留的設計：

- **切分由呼叫端傳進來**（`tr_idx` / `te_idx`），本檔不知道什麼是 test。
  超參搜尋只能拿到內層 validation 的索引，封存的外層 test 連傳都傳不進來
  （見 `nested.py` 的 `SealedTest`）。
- **每個 epoch 結束後可評估一次**（`eval_each_epoch`）。搜尋階段靠它做
  early stopping；正式評估階段關掉，只在最後評一次——因為「取最好的那個
  epoch」本身就是一次選擇，那個選擇要留在 validation 側，不能延伸到 test。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

from tiles import CLASSES

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Config:
    """一組完整的訓練設定。預設值＝README 第二節那組（見 `BASELINE`）。

    `weight_decay` 預設 1e-2 不是隨手填的：`train.py` 原本寫
    `torch.optim.AdamW(params, lr=LR)`，而 AdamW 的 weight_decay 預設就是
    1e-2——「沒有設定」在這裡不等於「沒有 weight decay」。要與既有數字對齊
    就必須把這個隱含值寫出來。
    """

    arch: str = "resnet18"
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 4
    freeze: str = "none"          # none｜backbone（線性探針）｜partial（末段＋頭）
    scheduler: str = "none"       # none｜cosine｜onecycle
    optimizer: str = "adamw"      # adamw｜sgd
    aug: str = "flips"            # flips｜flips_rot_jitter｜randaugment
    label_smoothing: float = 0.0
    img_size: int = 224
    batch: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE = Config()


# ---------------------------------------------------------------- 影像快取
# 以 (路徑, 尺寸) 為鍵：搜尋空間含 img_size，同一張圖可能要兩種解析度。
# 每個 epoch 重新解碼原圖會讓一輪從分鐘等級變成 40 分鐘以上（M1 16GB 實測）。
_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def preload(items: list[dict], img_size: int = 224) -> None:
    for it in items:
        key = (str(it["path"]), img_size)
        if key in _CACHE:
            continue
        img = Image.open(it["path"]).convert("RGB").resize((img_size, img_size))
        _CACHE[key] = torch.from_numpy(
            np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()


def cache_bytes() -> int:
    return sum(t.numel() for t in _CACHE.values())


# ---------------------------------------------------------------- 增強
# flips 這條路徑與 train.py 重構前完全相同（float→翻轉→正規化），
# tests/test_transforms_parity.py 釘住它；新增的兩條走 uint8→增強→轉浮點，
# 因為 RandAugment 的實作是針對 uint8 設計的。
def build_transforms(cfg: Config) -> tuple[Callable, Callable, bool]:
    """回傳 (訓練轉換, 評估轉換, 是否吃 uint8 輸入)。"""
    norm = v2.Normalize(mean=MEAN, std=STD)
    if cfg.aug == "flips":
        train = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                            v2.RandomVerticalFlip(p=0.5), norm])
        return train, v2.Compose([norm]), False

    to_float = v2.ToDtype(torch.float32, scale=True)
    if cfg.aug == "flips_rot_jitter":
        train = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            to_float, norm])
    elif cfg.aug == "randaugment":
        train = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=7), to_float, norm])
    else:
        raise ValueError(f"未知的增強設定：{cfg.aug}")
    return train, v2.Compose([to_float, norm]), True


class TileSet(Dataset):
    def __init__(self, items: list[dict], train: bool, cfg: Config, task: str):
        self.items, self.cfg, self.task = items, cfg, task
        tf_train, tf_eval, uint8_in = build_transforms(cfg)
        self.tf = tf_train if train else tf_eval
        self.uint8_in = uint8_in

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        x = _CACHE[(str(it["path"]), self.cfg.img_size)]
        if not self.uint8_in:
            x = x.float() / 255.0
        return self.tf(x), target_of(it, self.task)


def target_of(item: dict, task: str) -> int:
    """binary → 有瑕疵(1)／良品(0)；multiclass → CLASSES 的索引。"""
    return item["is_defect"] if task == "binary" else CLASSES.index(item["label"])


# ---------------------------------------------------------------- 模型
# 四個架構的分類頭掛在不同屬性上（resnet 是 .fc、mobilenet／efficientnet 是
# .classifier 裡的某一層），凍結時「末段」指的層也不同——這些差異不抽象掉，
# 直接列出來，因為抽象錯了不會報錯，只會靜默地凍到不該凍的地方。
ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (models.mobilenet_v3_small,
                           models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0,
                        models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_model(arch: str, n_out: int, freeze: str) -> nn.Module:
    fn, weights = ARCHS[arch]
    model = fn(weights=weights)

    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, n_out)
        head, tail = [model.fc], [model.layer4]
    elif arch == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_out)
        # head 只取最後那層 Linear：mobilenet 的 classifier 前面還有一層
        # Linear(576→1024)，整段解凍就有 0.59M 參數在訓練，那已經不是線性探針了。
        # 四個架構的 freeze="backbone" 必須是同一件事，否則對照組不對照。
        head, tail = [model.classifier[3]], [model.classifier[:3], model.features[-3:]]
    elif arch == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_out)
        head, tail = [model.classifier[1]], [model.features[-2:]]
    else:
        raise ValueError(f"未知架構：{arch}")

    if freeze == "none":
        return model
    for p in model.parameters():
        p.requires_grad = False
    unfrozen = head if freeze == "backbone" else head + tail
    for mod in unfrozen:
        for p in mod.parameters():
            p.requires_grad = True
    return model


def build_optimizer(model: nn.Module, cfg: Config):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"未知 optimizer：{cfg.optimizer}")


def build_scheduler(opt, cfg: Config, steps_per_epoch: int):
    total = max(1, steps_per_epoch * cfg.epochs)
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total)
    raise ValueError(f"未知 scheduler：{cfg.scheduler}")


# ---------------------------------------------------------------- 評估
def evaluate(model, dl, device, task: str, n_out: int) -> dict:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            p = torch.softmax(model(x.to(device)), dim=1)
            probs += p.cpu().tolist()
            ys += y.tolist()
    if task == "binary":
        p1 = [p[1] for p in probs]
        pred = [int(p >= 0.5) for p in p1]
        return {
            "test_defect_rate": round(float(np.mean(ys)), 4),
            "f1": round(f1_score(ys, pred, zero_division=0), 4),
            "auc": round(roc_auc_score(ys, p1), 4) if len(set(ys)) > 1 else None,
            "accuracy": round(float(np.mean([p == t for p, t in zip(pred, ys)])), 4),
        }
    pred = [int(np.argmax(p)) for p in probs]
    per_cls = f1_score(ys, pred, average=None, labels=range(n_out), zero_division=0)
    return {
        "macro_f1": round(float(f1_score(ys, pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float(np.mean([p == t for p, t in zip(pred, ys)])), 4),
        "per_class": {CLASSES[c]: {"f1": round(float(per_cls[c]), 4),
                                   "support": int(sum(1 for t in ys if t == c))}
                      for c in range(n_out)},
    }


def fit_eval(items: list[dict], tr_idx: list[int], te_idx: list[int], cfg: Config,
             seed: int, device, task: str = "binary",
             eval_each_epoch: bool = False, early_stop: str | None = None,
             patience: int = 3) -> dict:
    """訓練一次並評估。回傳最終指標；`eval_each_epoch` 時另附每個 epoch 的指標。

    `early_stop` 只在搜尋階段用（對象是內層 val）：連續 `patience` 個 epoch
    沒有更好就停，並回報最好的那個 epoch。正式評估階段一律不開——
    「取最好的 epoch」是一次選擇，選擇要留在 validation 側。

    RNG 消耗順序與重構前的 `train.py` 完全一致（manual_seed → 建模型 →
    建 DataLoader → 訓練），因為同種子下多抽或少抽一次 rand，整串訓練軌跡
    就會偏掉，而這不會有任何錯誤訊息。
    """
    torch.manual_seed(seed)
    tr = [items[i] for i in tr_idx]
    te = [items[i] for i in te_idx]
    n_out = 2 if task == "binary" else len(CLASSES)

    model = build_model(cfg.arch, n_out, cfg.freeze).to(device)
    opt = build_optimizer(model, cfg)
    # 類別不均（二分類：良品 952 vs 瑕疵 392）以反頻率加權，
    # 否則全猜良品就有 71%。
    w = torch.tensor([1.0 / max(1, sum(1 for x in tr if target_of(x, task) == c))
                      for c in range(n_out)], dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum(), label_smoothing=cfg.label_smoothing)

    dl_tr = DataLoader(TileSet(tr, True, cfg, task), batch_size=cfg.batch, shuffle=True)
    dl_te = DataLoader(TileSet(te, False, cfg, task), batch_size=cfg.batch)
    sched = build_scheduler(opt, cfg, len(dl_tr))

    if early_stop and not eval_each_epoch:
        raise ValueError("early stopping 需要 eval_each_epoch=True 才有可比的每輪指標")

    t0 = time.time()
    per_epoch, best, stale = [], -1.0, 0
    for ep in range(cfg.epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            lossf(model(x), y).backward()
            opt.step()
            if sched is not None:
                sched.step()
        if eval_each_epoch:
            per_epoch.append({"epoch": ep + 1} | evaluate(model, dl_te, device, task, n_out))
            model.train()
            if early_stop:
                cur = per_epoch[-1].get(early_stop) or -1.0
                stale = 0 if cur > best else stale + 1
                best = max(best, cur)
                if stale >= patience:
                    break
    train_sec = round(time.time() - t0, 1)

    out = {"seed": seed, "task": task, "n_train": len(tr), "n_test": len(te),
           "train_sec": train_sec, "epochs_run": len(per_epoch) or cfg.epochs}
    out |= per_epoch[-1] if per_epoch else evaluate(model, dl_te, device, task, n_out)
    out.pop("epoch", None)
    if eval_each_epoch:
        out["per_epoch"] = per_epoch
        # early stopping 的回報對象是「最好的那個 epoch」，不是最後一個。
        bi = max(range(len(per_epoch)),
                 key=lambda i: per_epoch[i].get(early_stop or "f1") or -1.0)
        out["best_epoch"] = per_epoch[bi]["epoch"]
        out["best_epoch_metrics"] = {k: v for k, v in per_epoch[bi].items() if k != "epoch"}
    return out


def with_epochs(cfg: Config, epochs: int) -> Config:
    return replace(cfg, epochs=epochs)


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
