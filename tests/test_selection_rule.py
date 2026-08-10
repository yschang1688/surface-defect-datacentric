"""挑超參的規則本身要被釘住：規則若能事後調整，搜尋就只是在挑形容詞。

`analyze_results.py` 的可宣稱門檻寫死是同一個道理——差別在那裡管的是
「差距可不可以宣稱」，這裡管的是「哪一組設定勝出」。搜尋跑完才決定
「要不要看訓練成本」「同分帶要多寬」，等於在結果出來後選一個自己喜歡的贏家。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from search import SELECTION_RULE, TIE_BAND_SD_MULT, select  # noqa: E402


def row(idx: int, mean: float, sd: float = 0.02, secs: float = 60.0,
        epochs: int = 4) -> dict:
    return {"idx": idx, "val_f1_mean": mean, "val_f1_sd": sd,
            "train_sec_median": secs, "n_folds": 3, "config": {"epochs": epochs}}


def test_cheapest_wins_inside_the_tie_band() -> None:
    """帶內名次是雜訊，拿雜訊當理由選一個貴三倍的模型，代價卻是真的。"""
    rows = [row(0, 0.800, secs=300), row(1, 0.795, secs=60), row(2, 0.700, secs=10)]
    sel = select(rows)
    assert sel["best_by_mean"] == 0
    assert set(sel["tie_set"]) == {0, 1}, "0.005 的差距落在同分帶內，兩者應同列"
    assert sel["winner"] == 1, "同分帶內應取訓練時間最短者"


def test_clearly_better_config_wins_even_if_expensive() -> None:
    """探針：若『挑最便宜』蓋過一切，這條會紅——那規則就不是在挑好設定了。"""
    rows = [row(0, 0.900, sd=0.005, secs=300), row(1, 0.700, sd=0.005, secs=10)]
    sel = select(rows)
    assert sel["tie_set"] == [0]
    assert sel["winner"] == 0


def test_tie_band_widens_with_fold_noise() -> None:
    """折間浮動越大，能宣稱的名次差就越少——這是雜訊帶的定義。"""
    quiet = select([row(0, 0.80, sd=0.001), row(1, 0.78, sd=0.001)])
    noisy = select([row(0, 0.80, sd=0.100), row(1, 0.78, sd=0.100)])
    assert quiet["tie_band"] < noisy["tie_band"]
    assert quiet["tie_set"] == [0], "雜訊小時 0.02 的差距應該分得出來"
    assert set(noisy["tie_set"]) == {0, 1}, "雜訊大時同樣的差距不可宣稱"


def test_ties_break_deterministically() -> None:
    """完全同分同價時必須每次挑同一個，否則重跑會換贏家。"""
    rows = [row(3, 0.80), row(1, 0.80), row(2, 0.80)]
    assert select(rows)["winner"] == select(list(reversed(rows)))["winner"] == 1


def test_rule_constants_are_pinned() -> None:
    """規則被放寬時，這條會紅——改動必須是有意識的，且要一起改測試。"""
    assert TIE_BAND_SD_MULT == 2.0
    assert "訓練時間中位數最小者" in SELECTION_RULE
    assert "best_mean − 2.0 × 折間標準誤" in SELECTION_RULE


def test_selection_is_recorded_in_output() -> None:
    """規則字串要跟著結果落檔，事後才對得起來。"""
    sel = select([row(0, 0.8), row(1, 0.7)])
    assert sel["rule"] == SELECTION_RULE
    for k in ("pooled_fold_sd", "tie_band", "tie_set", "winner", "best_by_mean"):
        assert k in sel


@pytest.mark.parametrize("n", [1, 2, 5])
def test_select_survives_small_tables(n: int) -> None:
    sel = select([row(i, 0.8 - 0.01 * i) for i in range(n)])
    assert sel["winner"] in range(n)
