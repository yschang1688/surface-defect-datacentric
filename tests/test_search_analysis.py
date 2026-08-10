"""邊際分析的判定：樣本太薄時必須說「不判定」，而不是給一個看起來正常的結論。

n<3 的堆用 Welch 檢定會回 nan，而 `nan < 0.10` 是 False——判定函式會靜默
落到「落在雜訊帶內」。那句話讀起來完全正常，實際上背後只有一組設定撐著。
這種假結論不會有錯誤訊息，只能靠測試釘住。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyze_search import marginals  # noqa: E402


def row(idx: int, mean: float, **cfg) -> dict:
    base = {"arch": "resnet18", "lr": 3e-4, "weight_decay": 0.0, "epochs": 4,
            "freeze": "none", "scheduler": "none", "optimizer": "adamw",
            "aug": "flips", "label_smoothing": 0.0, "img_size": 224, "batch": 32}
    return {"idx": idx, "val_f1_mean": mean, "config": base | cfg}


def test_thin_buckets_are_not_judged() -> None:
    rows = [row(0, 0.80, freeze="none"), row(1, 0.60, freeze="backbone")]
    m = marginals(rows, band=0.05)
    assert m["freeze"]["p_value"] is None
    assert "不判定" in m["freeze"]["verdict"], (
        "每堆只有 1 組就給判定——nan 的 p 值會被讀成『落在雜訊帶內』")


def test_a_real_difference_is_judged_when_buckets_are_thick() -> None:
    """探針：若所有東西都被判成『不判定』，上一條的綠沒有意義。"""
    rows = ([row(i, 0.80 + 0.001 * i, freeze="none") for i in range(5)]
            + [row(10 + i, 0.50 + 0.001 * i, freeze="backbone") for i in range(5)])
    m = marginals(rows, band=0.05)
    assert m["freeze"]["p_value"] is not None
    assert m["freeze"]["verdict"] == "可宣稱"
    assert m["freeze"]["spread"] > 0.2


def test_stable_but_tiny_difference_is_not_enough_to_pick_a_setting() -> None:
    """堆間差異可能既真實又不值得——p 小與「大於同分帶」是兩件事。

    這裡的 0.01 差距在堆內浮動極小時 p < 0.05（確實不同），但比一次重跑的
    浮動還小（同分帶 0.05）。只講 p 會得到「可宣稱」，據此換設定卻換不到東西。
    """
    rows = ([row(i, 0.80 + 0.002 * (i % 3), freeze="none") for i in range(5)]
            + [row(10 + i, 0.79 + 0.002 * (i % 3), freeze="backbone") for i in range(5)])
    m = marginals(rows, band=0.05)
    assert m["freeze"]["p_value"] < 0.05
    assert m["freeze"]["verdict"] == "統計上可分辨，但差距小於同分帶，不足以據此選設定"


def test_multivalue_dimension_uses_the_tie_band_not_a_p_value() -> None:
    """三個取值以上不報 p：多重比較不校正就報 p 會製造假訊號。"""
    rows = [row(i, 0.7 + 0.01 * (i % 3), epochs=[4, 6, 8][i % 3]) for i in range(9)]
    m = marginals(rows, band=0.05)
    assert m["epochs"]["p_value"] is None
    assert "同分帶" in m["epochs"]["verdict"]
