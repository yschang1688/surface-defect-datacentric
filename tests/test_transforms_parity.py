"""增強改用 torchvision.transforms.v2 之後，行為必須與手寫版逐位元組相同。

換實作是為了讓程式碼與對外說法一致（履歷／JD 對話講 torchvision 影像前處理與
增強，原始碼卻只 import 了 torchvision.models）。但**換實作不該換結果**——
results/ 裡既有的 F1 0.8801→0.7607 全部產於手寫版，若新版在同種子下給出不同
張量，那些數字就不再可重現，而本專案的全部結論都建立在那些數字上。

所以這裡保留舊實作當對照組，比對三件事：
1. 同種子下 train 分支的輸出張量完全相同（翻轉與否、翻哪個軸、正規化）
2. RNG 消耗量相同——若 v2 多抽或少抽一次 rand，之後所有種子行為都會錯位，
   而這不會有任何錯誤訊息，只會讓「重跑對不上」變成一個查不到原因的現象
3. eval 分支不做增強，只正規化
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import EVAL_TF, MEAN, STD, TRAIN_TF  # noqa: E402


def legacy_train(x: torch.Tensor) -> torch.Tensor:
    """重構前的手寫實作，原封不動保留作為對照。"""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])   # 水平
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[1])   # 垂直
    return (x - mean) / std


def legacy_eval(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std


def sample(seed: int = 0) -> torch.Tensor:
    # 非對稱內容：全 0 或對稱圖樣會讓「翻轉了沒」測不出來
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, 8, 8, generator=g)


@pytest.mark.parametrize("seed", range(12))
def test_train_transform_matches_legacy(seed: int) -> None:
    x = sample(seed)
    torch.manual_seed(seed)
    got = TRAIN_TF(x.clone())
    torch.manual_seed(seed)
    want = legacy_train(x.clone())
    assert torch.equal(got, want), (
        f"seed={seed}：v2 版與手寫版輸出不同。可能是翻轉機率、翻轉軸或順序變了——"
        "results/ 既有數字產於手寫版，行為一變那些數字就不可重現。")


@pytest.mark.parametrize("seed", range(12))
def test_rng_consumption_matches_legacy(seed: int) -> None:
    """RNG 抽用次數必須一致，否則同種子的後續行為整串錯位（且不會報錯）。"""
    x = sample(seed)
    torch.manual_seed(seed)
    TRAIN_TF(x.clone())
    after_v2 = torch.rand(1).item()
    torch.manual_seed(seed)
    legacy_train(x.clone())
    after_legacy = torch.rand(1).item()
    assert after_v2 == after_legacy, (
        f"seed={seed}：兩版消耗的亂數個數不同（v2 後續值 {after_v2}、"
        f"手寫版 {after_legacy}），同種子的訓練軌跡會整串偏掉。")


def test_eval_transform_is_normalise_only() -> None:
    x = sample(3)
    assert torch.equal(EVAL_TF(x.clone()), legacy_eval(x.clone()))


def test_train_transform_actually_flips_sometimes() -> None:
    """探針：若增強被誤關成 no-op，上面的比對測試會因兩邊同時 no-op 而假綠。"""
    x = sample(1)
    flipped = 0
    for seed in range(30):
        torch.manual_seed(seed)
        if not torch.equal(TRAIN_TF(x.clone()), EVAL_TF(x.clone())):
            flipped += 1
    assert 5 <= flipped <= 25, (
        f"30 次裡只有 {flipped} 次發生翻轉——p=0.5 的兩軸翻轉理應約 22 次（1−0.25）。"
        "落在區間外代表增強沒真的接上，或機率被改掉了。")
