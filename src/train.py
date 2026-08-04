"""瑕疵二分類（有瑕疵／良品）：兩種切分協定 × 多顆種子。

本檔要回答的不是「準確率多少」，而是**這個準確率能不能相信**：

1. **隨機切分 vs 分組留出**。同一片磁磚的多張影像若散落在訓練與測試兩側，
   模型只要記住這片磁磚長什麼樣就能答對——量到的是記憶力不是泛化力。
   分組留出把同一片磁磚整組關進同一側，量到的才是「沒看過的磁磚」。
2. **雜訊帶**。同一協定換種子重跑多次，指標本身就會浮動。兩個數字的差
   若小於這條浮動帶，就不可以宣稱誰比較好——這是量測不確定度的常識，
   在模型評估裡卻常被忽略。

模型不是重點：ResNet-18 ImageNet 預訓練 + 線性頭微調，兩種協定完全同一套
超參數與訓練輪數，唯一的差別是資料怎麼切。
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

from tiles import group_ids, list_images

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


class TileSet(Dataset):
    """train=True 時做翻轉增強；正規化兩者相同。"""

    def __init__(self, items: list[dict], train: bool):
        self.items, self.train = items, train
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        x = _CACHE[str(it["path"])].float() / 255.0
        if self.train:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])   # 水平
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])   # 垂直
        return (x - self.mean) / self.std, it["is_defect"]


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


def run_once(items, gid, protocol: str, seed: int, device) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = SPLITS[protocol](items, gid, rng)
    tr = [items[i] for i in tr_idx]
    te = [items[i] for i in te_idx]

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    # 類別不均（良品 952 vs 瑕疵 392）：以反頻率加權，否則全猜良品就有 71%
    w = torch.tensor([1.0 / max(1, sum(1 for x in tr if x["is_defect"] == c))
                      for c in (0, 1)], dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum())

    dl_tr = DataLoader(TileSet(tr, True), batch_size=BATCH, shuffle=True)
    dl_te = DataLoader(TileSet(te, False), batch_size=BATCH)

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
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            probs += p.cpu().tolist()
            ys += y.tolist()
    pred = [int(p >= 0.5) for p in probs]
    return {
        "protocol": protocol, "seed": seed,
        "n_train": len(tr), "n_test": len(te),
        "test_defect_rate": round(float(np.mean(ys)), 4),
        "f1": round(f1_score(ys, pred, zero_division=0), 4),
        "auc": round(roc_auc_score(ys, probs), 4) if len(set(ys)) > 1 else None,
        "accuracy": round(float(np.mean([p == t for p, t in zip(pred, ys)])), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out", default="training_results.json")
    ap.add_argument("--protocols", nargs="+", default=["random", "grouped"])
    a = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    items = list_images()
    gid = group_ids(items)
    preload(items)
    print(f"{len(items)} 張影像 → {gid.max() + 1} 個磁磚分組｜裝置 {device}（影像已快取）")

    rows, t0 = [], time.time()
    for protocol in a.protocols:
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            r = run_once(items, gid, protocol, seed, device)
            rows.append(r)
            print(f"  {protocol:8s} seed={seed} F1={r['f1']:.4f} AUC={r['auc']} "
                  f"acc={r['accuracy']:.4f}（測試 {r['n_test']} 張，瑕疵率 {r['test_defect_rate']:.2f}）")

    summary = {}
    for protocol in a.protocols:
        v = [r for r in rows if r["protocol"] == protocol]
        for metric in ("f1", "auc", "accuracy"):
            xs = [r[metric] for r in v if r[metric] is not None]
            summary.setdefault(protocol, {})[metric] = {
                "mean": round(float(np.mean(xs)), 4),
                "sd": round(float(np.std(xs, ddof=1)), 4),
                "min": round(float(np.min(xs)), 4),
                "max": round(float(np.max(xs)), 4),
            }
    out = {"seeds": a.seeds, "epochs": EPOCHS, "groups": int(gid.max() + 1),
           "elapsed_sec": round(time.time() - t0, 1), "runs": rows, "summary": summary}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / a.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
