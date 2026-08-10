"""瑕疵分類（二分類／六類擇一）：兩種切分協定 × 多顆種子。

本檔要回答的不是「準確率多少」，而是**這個準確率能不能相信**：

1. **隨機切分 vs 分組留出**。同一片磁磚的多張影像若散落在訓練與測試兩側，
   模型只要記住這片磁磚長什麼樣就能答對——量到的是記憶力不是泛化力。
   分組留出把同一片磁磚整組關進同一側，量到的才是「沒看過的磁磚」。
2. **雜訊帶**。同一協定換種子重跑多次，指標本身就會浮動。兩個數字的差
   若小於這條浮動帶，就不可以宣稱誰比較好——這是量測不確定度的常識，
   在模型評估裡卻常被忽略。

模型不是重點：ResNet-18 ImageNet 預訓練 + 線性頭微調，兩種協定完全同一套
超參數與訓練輪數，唯一的差別是資料怎麼切。

任務有兩種（`--task`），**協定的結論不因任務而改變**，切分才是本專案的主題：

- `binary`（預設）：有瑕疵／良品。README 與 results/ 既有數字全部產於此設定，
  改動預設值會讓那些數字對不上，所以預設不動。
- `multiclass`：資料集原本的六類（Blowhole／Break／Crack／Fray／Uneven／Free）。
  **這個設定的類別級結論本來就薄**——Fray 僅 32 張，切四分之一去測試只剩個位數，
  單顆種子的類別級 F1 幾乎全是抽樣雜訊。所以輸出一律附 `per_class` 的 support，
  讓「這格數字有幾張撐著」和數字本身一起被看到；support 個位數的類別不得拿來宣稱。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

from tiles import CLASSES, group_ids, list_images

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMG_SIZE = 224
EPOCHS = 4
BATCH = 32
LR = 3e-4
TEST_FRAC = 0.25

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# 影像先一次解碼並縮到目標尺寸，以 uint8 存在記憶體（1,344×3×224×224 ≈ 200 MB）。
# 原本每個 epoch 重讀＋重解原圖，在記憶體吃緊的機器上一輪要跑 40 分鐘以上；
# 快取後每輪降到分鐘等級。訓練邏輯完全不變，只是不再重複做同一件解碼工作。
_CACHE: dict[str, torch.Tensor] = {}


def preload(items: list[dict]) -> None:
    if _CACHE:
        return
    for it in items:
        img = Image.open(it["path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        _CACHE[str(it["path"])] = torch.from_numpy(
            np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()


# 增強與正規化改用 torchvision.transforms.v2（原為手寫 torch.flip ＋ 手算標準化）。
# 換的是實作不是行為：v2.RandomHorizontalFlip(p) 內部同樣抽一次 torch.rand(1)、
# 同樣在 < p 時翻轉，順序也相同（水平→垂直→正規化），所以同種子下輸出張量
# 與舊版逐位元組相同——tests/test_transforms_parity.py 就是釘住這件事的，
# 它保留了舊版的手寫實作當對照，兩者不一致即紅。
#
# 為什麼要換：手寫版能跑，但履歷與 JD 對話裡講的是「torchvision 影像前處理與增強」，
# 而原始碼只 import 了 torchvision.models——**說法與程式碼不一致**。
# 對齊的方向是改程式碼，不是改說法。
TRAIN_TF = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.Normalize(mean=MEAN, std=STD),
])
EVAL_TF = v2.Compose([v2.Normalize(mean=MEAN, std=STD)])


def target_of(item: dict, task: str) -> int:
    """binary → 有瑕疵(1)／良品(0)；multiclass → CLASSES 的索引。"""
    return item["is_defect"] if task == "binary" else CLASSES.index(item["label"])


class TileSet(Dataset):
    """train=True 時做翻轉增強；正規化兩者相同。"""

    def __init__(self, items: list[dict], train: bool, task: str = "binary"):
        self.items, self.train, self.task = items, train, task
        self.tf = TRAIN_TF if train else EVAL_TF

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        x = _CACHE[str(it["path"])].float() / 255.0
        return self.tf(x), target_of(it, self.task)


def split_random(items, gid, rng) -> tuple[list[int], list[int]]:
    idx = rng.permutation(len(items))
    cut = int(len(items) * TEST_FRAC)
    return idx[cut:].tolist(), idx[:cut].tolist()


def split_grouped(items, gid, rng) -> tuple[list[int], list[int]]:
    """整組進同一側；依組別抽樣直到測試集達到目標比例。"""
    groups = rng.permutation(gid.max() + 1)
    target, test_groups, n = int(len(items) * TEST_FRAC), set(), 0
    for g in groups:
        if n >= target:
            break
        test_groups.add(int(g))
        n += int((gid == g).sum())
    test = [i for i in range(len(items)) if gid[i] in test_groups]
    train = [i for i in range(len(items)) if gid[i] not in test_groups]
    return train, test


SPLITS = {"random": split_random, "grouped": split_grouped}


def run_once(items, gid, protocol: str, seed: int, device, task: str = "binary") -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = SPLITS[protocol](items, gid, rng)
    tr = [items[i] for i in tr_idx]
    te = [items[i] for i in te_idx]
    n_out = 2 if task == "binary" else len(CLASSES)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_out)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    # 類別不均（二分類：良品 952 vs 瑕疵 392；六類更懸殊，Fray 僅 32 張）：
    # 以反頻率加權，否則二分類全猜良品就有 71%、六類全猜 Free 也有同一個數量級。
    w = torch.tensor([1.0 / max(1, sum(1 for x in tr if target_of(x, task) == c))
                      for c in range(n_out)], dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum())

    dl_tr = DataLoader(TileSet(tr, True, task), batch_size=BATCH, shuffle=True)
    dl_te = DataLoader(TileSet(te, False, task), batch_size=BATCH)

    model.train()
    for _ in range(EPOCHS):
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            lossf(model(x), y).backward()
            opt.step()

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in dl_te:
            p = torch.softmax(model(x.to(device)), dim=1)
            probs += p.cpu().tolist()
            ys += y.tolist()

    base = {"protocol": protocol, "seed": seed, "task": task,
            "n_train": len(tr), "n_test": len(te)}
    if task == "binary":
        p1 = [p[1] for p in probs]
        pred = [int(p >= 0.5) for p in p1]
        return base | {
            "test_defect_rate": round(float(np.mean(ys)), 4),
            "f1": round(f1_score(ys, pred, zero_division=0), 4),
            "auc": round(roc_auc_score(ys, p1), 4) if len(set(ys)) > 1 else None,
            "accuracy": round(float(np.mean([p == t for p, t in zip(pred, ys)])), 4),
        }

    pred = [int(np.argmax(p)) for p in probs]
    per_cls = f1_score(ys, pred, average=None, labels=range(n_out), zero_division=0)
    return base | {
        # macro 平均刻意不加權：小類別（Fray）在 macro 下與大類別等重，
        # 這正是要看見的——用 weighted 會被 Free 的大 support 蓋掉真正的失敗處。
        "macro_f1": round(float(f1_score(ys, pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float(np.mean([p == t for p, t in zip(pred, ys)])), 4),
        "per_class": {CLASSES[c]: {"f1": round(float(per_cls[c]), 4),
                                   "support": int(sum(1 for t in ys if t == c))}
                      for c in range(n_out)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out", default="training_results.json")
    ap.add_argument("--protocols", nargs="+", default=["random", "grouped"])
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary",
                    help="binary＝有瑕疵／良品（預設，README 數字產於此）；"
                         "multiclass＝資料集原本的六類（Fray 僅 32 張，類別級結論薄）")
    a = ap.parse_args()
    if a.task == "multiclass" and a.out == "training_results.json":
        # 預設檔名會被 analyze_results.py 撿走與二分類結果混算，直接擋掉
        a.out = "training_results_multiclass.json"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    items = list_images()
    gid = group_ids(items)
    preload(items)
    print(f"{len(items)} 張影像 → {gid.max() + 1} 個磁磚分組｜裝置 {device}（影像已快取）")

    metrics = ("f1", "auc", "accuracy") if a.task == "binary" else ("macro_f1", "accuracy")
    rows, t0 = [], time.time()
    for protocol in a.protocols:
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            r = run_once(items, gid, protocol, seed, device, a.task)
            rows.append(r)
            if a.task == "binary":
                print(f"  {protocol:8s} seed={seed} F1={r['f1']:.4f} AUC={r['auc']} "
                      f"acc={r['accuracy']:.4f}（測試 {r['n_test']} 張，"
                      f"瑕疵率 {r['test_defect_rate']:.2f}）")
            else:
                thin = [c for c, v in r["per_class"].items() if v["support"] < 10]
                print(f"  {protocol:8s} seed={seed} macroF1={r['macro_f1']:.4f} "
                      f"acc={r['accuracy']:.4f}（測試 {r['n_test']} 張）"
                      + (f"｜support <10 的類別：{'、'.join(thin)}" if thin else ""))

    summary = {}
    for protocol in a.protocols:
        v = [r for r in rows if r["protocol"] == protocol]
        for metric in metrics:
            xs = [r[metric] for r in v if r.get(metric) is not None]
            summary.setdefault(protocol, {})[metric] = {
                "mean": round(float(np.mean(xs)), 4),
                "sd": round(float(np.std(xs, ddof=1)), 4),
                "min": round(float(np.min(xs)), 4),
                "max": round(float(np.max(xs)), 4),
            }
    if a.task == "multiclass":
        # 各類別的 support 跨種子加總：讓「這格 F1 有幾張撐著」和 F1 同時落檔，
        # 否則一個 0.00 的類別級 F1 看起來像模型爛，實際上是測試集只有 6 張。
        summary["per_class_support_total"] = {
            c: sum(r["per_class"][c]["support"] for r in rows) for c in CLASSES}
    out = {"seeds": a.seeds, "epochs": EPOCHS, "task": a.task,
           "groups": int(gid.max() + 1),
           "elapsed_sec": round(time.time() - t0, 1), "runs": rows, "summary": summary}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / a.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
