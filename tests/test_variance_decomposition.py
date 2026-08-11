"""變異拆解：同分帶太寬時，要先問「這個浮動是誰造成的」。

處方完全不同——折主效應是所有候選共有的加法偏移，成對比較消得掉一部分；
殘差才是真雜訊，只能靠加折數。把兩者混在一起看，會挑錯要改的東西
（本專案就是：以為換成成對估就好，實測只消掉 38%，仍會挑錯）。

所以這裡用**答案已知的合成矩陣**釘住拆解的數學，而不是相信實測輸出。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyze_search import decompose, folds_needed  # noqa: E402


def rows_from(M: np.ndarray) -> list[dict]:
    return [{"idx": i,
             "val_f1_mean": round(float(r.mean()), 4),
             "val_f1_sd": round(float(r.std(ddof=1)), 4),
             "folds": [{"fold": k, "f1": float(v)} for k, v in enumerate(r)]}
            for i, r in enumerate(M)]


def test_pure_fold_effect_is_attributed_to_folds() -> None:
    """所有設定共用同一組折偏移、彼此無差異 → 折主效應應接近 100%。"""
    offsets = np.array([-0.1, 0.0, 0.1])
    M = np.tile(0.7 + offsets, (8, 1))
    d = decompose(rows_from(M))
    assert d["share_fold_effect"] > 0.99
    assert d["share_config_effect"] < 0.01


def test_pure_config_effect_is_attributed_to_configs() -> None:
    """探針：反過來，設定之間有差、折之間沒差 → 設定主效應應接近 100%。

    沒有這條的話，一個永遠回「折效應 100%」的壞實作也會讓上一條通過。
    """
    M = np.tile(np.linspace(0.4, 0.9, 8).reshape(-1, 1), (1, 3))
    d = decompose(rows_from(M))
    assert d["share_config_effect"] > 0.99
    assert d["share_fold_effect"] < 0.01


def test_pairing_removes_the_shared_fold_offset() -> None:
    """折偏移是共有的加法項，兩兩相減必然消掉——這是成對估的全部理由。"""
    base = np.linspace(0.4, 0.9, 6).reshape(-1, 1)
    M = base + np.array([-0.1, 0.0, 0.1])          # 每個設定都吃同一組偏移
    d = decompose(rows_from(M))
    assert d["paired_diff_sd_median"] == pytest.approx(0.0, abs=1e-9)
    assert d["variance_removed_by_pairing"] > 0.99


def test_residual_noise_survives_pairing() -> None:
    """探針：純殘差雜訊時，成對**不該**消掉它——否則上一條的綠是假的。"""
    rng = np.random.default_rng(0)
    M = 0.7 + rng.normal(0, 0.05, size=(30, 3))
    d = decompose(rows_from(M))
    assert d["variance_removed_by_pairing"] < 0.35, (
        "成對居然消掉大部分的純雜訊——那代表它消的不是折偏移，實作有問題")


def test_folds_needed_scales_with_the_square_of_the_ratio() -> None:
    """帶寬＝2·sd/√k：sd 翻倍要四倍折數。

    邊界一律進位一折——同分帶用 `>= best − band`，剛好等於帶寬的仍算同分，
    所以「剛好相等」不算落到帶外。
    """
    assert folds_needed(paired_sd=0.05, gap=0.10) == 2      # x=1 恰好整除 → 再加一折
    assert folds_needed(paired_sd=0.10, gap=0.10) == 5      # x=4 恰好整除 → 5
    assert folds_needed(paired_sd=0.20, gap=0.10) == 17     # x=16 → 17
    # 單調性探針：雜訊變大，需要的折數不得變少（小幅變動會因進位而持平，故取大步）
    assert folds_needed(paired_sd=0.09, gap=0.10) > folds_needed(paired_sd=0.05, gap=0.10)


def test_folds_needed_matches_the_recorded_case() -> None:
    """實測案例：成對 sd 0.063、真差距 0.065 → 3 折不夠，需 5–6 折。"""
    assert folds_needed(0.063, 0.0651) == 4    # 用全體中位數
    assert folds_needed(0.0726, 0.0651) == 5   # 用該對自己的逐折差 sd
