"""訓練迴圈搬進 experiment.py 之後，BASELINE 必須重現 results/ 既有數字。

搬家的動機是共用：超參搜尋若跑在另一套訓練實作上，搜出來的贏家可能贏在
實作差異上，而不是超參上。但**搬家不該換結果**——README 的
0.8801 → 0.7607、p = 0.0006 全部產於搬家前的 `train.py`，若同一顆種子現在
給出不同的 F1，那些數字就不再可重現，整份結論跟著失效。

這裡分兩層釘：

1. **靜態層**（CI 就跑得動，不需要資料集與 GPU）：`BASELINE` 的每一個欄位
   必須等於搬家前寫死在 `train.py` 裡的值。最容易錯的是 `weight_decay`——
   舊碼寫 `AdamW(params, lr=LR)`，看起來「沒設 weight decay」，實際上
   AdamW 的預設值是 1e-2。把它讀成 0 會讓 BASELINE 悄悄變成另一組超參，
   而分數只差一點點，看起來完全正常。
2. **實跑層**（需要資料集，本機跑）：`--runtime-parity` 標記的測試會實際訓練
   兩顆種子，逐位元組比對 results/training_results.json 裡的同一列。
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiment import BASELINE  # noqa: E402

# 搬家前 train.py 的常數與 AdamW／CrossEntropyLoss 的實際預設值。
FROZEN_BASELINE = {
    "arch": "resnet18", "lr": 3e-4, "weight_decay": 1e-2, "epochs": 4,
    "freeze": "none", "scheduler": "none", "optimizer": "adamw",
    "aug": "flips", "label_smoothing": 0.0, "img_size": 224, "batch": 32,
}


def test_baseline_matches_the_frozen_recipe() -> None:
    assert BASELINE.to_dict() == FROZEN_BASELINE, (
        "BASELINE 與搬家前的設定不同——results/ 既有數字全部產於後者，"
        "改動前請先重跑並更新 README。")


def test_train_py_constants_still_read_from_baseline() -> None:
    """train.py 若又把常數寫死回去，兩邊就會各自漂移。"""
    src = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
    for name in ("IMG_SIZE", "EPOCHS", "BATCH", "LR"):
        assert re.search(rf"^{name} = BASELINE\.", src, re.M), (
            f"{name} 不再取自 BASELINE，train.py 與 experiment.py 會各走各的。")


def test_adamw_default_weight_decay_is_really_1e_2() -> None:
    """探針：上面那條的前提是「AdamW 預設 wd=1e-2」。前提若被 torch 改掉，
    FROZEN_BASELINE 就寫錯了，而測試仍會全綠——所以直接對 torch 問一次。"""
    import inspect

    import torch

    default = inspect.signature(torch.optim.AdamW).parameters["weight_decay"].default
    assert default == 1e-2, (
        f"這個版本的 AdamW 預設 weight_decay 是 {default}，不是 1e-2——"
        "搬家前 `AdamW(params, lr=LR)` 的實際行為因此不同於 FROZEN_BASELINE。")


@pytest.mark.runtime_parity
@pytest.mark.parametrize("protocol,seed", [("grouped", 0), ("random", 0)])
def test_runtime_matches_recorded_run(protocol: str, seed: int) -> None:
    """實跑比對：需要資料集，預設不在 CI 跑（`-m runtime_parity` 才會選到）。"""
    recorded = None
    for f in sorted((ROOT / "results").glob("training_results*.json")):
        for r in json.loads(f.read_text(encoding="utf-8"))["runs"]:
            if r.get("task", "binary") == "binary" and r["protocol"] == protocol \
                    and r["seed"] == seed:
                recorded = r
                break
        if recorded:
            break
    assert recorded, "找不到可比對的既有紀錄"

    import numpy as np
    import torch
    from tiles import group_ids, list_images
    from train import SPLITS, preload

    items = list_images()
    gid = group_ids(items)
    preload(items)
    from experiment import fit_eval
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tr, te = SPLITS[protocol](items, gid, np.random.default_rng(seed))
    got = fit_eval(items, tr, te, BASELINE, seed, device, "binary")
    for m in ("f1", "auc", "accuracy", "n_train", "n_test"):
        assert got[m] == recorded[m], (
            f"{protocol} seed={seed} 的 {m}：重構後 {got[m]}、既有紀錄 {recorded[m]}。"
            "訓練軌跡已經偏掉，results/ 的數字不再可重現。")
