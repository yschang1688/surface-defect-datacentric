"""品質閘注入測試：閘門在真實資料上零觸發，所以它會不會響必須另外證明。

「跑過、沒有發現問題」與「根本檢查不出問題」在輸出上長得一模一樣，
差別只有靠植入已知缺陷才看得出來。
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quality_gate import check  # noqa: E402


def save(tmp_path: Path, name: str, arr: np.ndarray) -> Path:
    p = tmp_path / f"{name}.jpg"
    Image.fromarray(arr.astype("uint8")).save(p, quality=95)
    return p


@pytest.fixture
def normal(tmp_path):
    """基準影像：尺寸、曝光、對比、銳利度都在正常範圍。"""
    rng = np.random.default_rng(0)
    a = rng.normal(128, 40, (300, 200)).clip(0, 255)
    return save(tmp_path, "normal", a)


def test_normal_image_passes(normal):
    assert check(normal, "Free").verdict == "PASS"


def test_undexposed_rejected(tmp_path):
    a = np.random.default_rng(0).normal(8, 3, (300, 200)).clip(0, 255)
    v = check(save(tmp_path, "dark", a), "Free")
    assert v.verdict == "REJECT" and any("UNDEREXPOSED" in r for r in v.rules)


def test_overexposed_rejected(tmp_path):
    a = np.random.default_rng(0).normal(248, 3, (300, 200)).clip(0, 255)
    v = check(save(tmp_path, "bright", a), "Free")
    assert v.verdict == "REJECT" and any("OVEREXPOSED" in r for r in v.rules)


def test_too_small_rejected(tmp_path):
    a = np.random.default_rng(0).normal(128, 40, (300, 32)).clip(0, 255)
    v = check(save(tmp_path, "small", a), "Free")
    assert v.verdict == "REJECT" and any("SIZE_TOO_SMALL" in r for r in v.rules)


def test_extreme_aspect_rejected(tmp_path):
    a = np.random.default_rng(0).normal(128, 40, (700, 70)).clip(0, 255)
    v = check(save(tmp_path, "long", a), "Free")
    assert v.verdict == "REJECT" and any("EXTREME_ASPECT" in r for r in v.rules)


def test_low_contrast_is_conditional_not_reject(tmp_path):
    """三級判定的意義：可疑但可用的資料不該被當成廢料丟掉。"""
    a = np.random.default_rng(0).normal(128, 3, (300, 200)).clip(0, 255)
    v = check(save(tmp_path, "flat", a), "Free")
    assert v.verdict == "CONDITIONAL" and any("LOW_CONTRAST" in r for r in v.rules)


def test_blur_is_conditional(tmp_path):
    """離焦：由平滑漸層構成，對比夠但沒有高頻紋理。"""
    y = np.linspace(60, 200, 300)[:, None]
    x = np.linspace(0, 30, 200)[None, :]
    v = check(save(tmp_path, "blur", (y + x).clip(0, 255)), "Free")
    assert v.verdict == "CONDITIONAL" and any("POSSIBLY_BLURRED" in r for r in v.rules)


def test_hard_rule_wins_over_soft(tmp_path):
    """同時命中硬規則與軟規則時，判定必須是 REJECT——分級不可被稀釋。"""
    a = np.full((300, 200), 5.0)      # 過暗（硬）＋ 對比為零（軟）
    v = check(save(tmp_path, "both", a), "Free")
    assert v.verdict == "REJECT"
    assert any("UNDEREXPOSED" in r for r in v.rules)
    assert any("LOW_CONTRAST" in r for r in v.rules)
