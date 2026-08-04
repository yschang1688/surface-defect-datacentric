"""切分完整性測試：分組留出若沒真的隔離，整個專案的結論就是空的。

這裡的測試對象不是模型，是**評估協定本身**——評估協定出錯不會有任何
錯誤訊息，只會給你一個好看的數字。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tiles import group_ids  # noqa: E402
from train import split_grouped, split_random  # noqa: E402


def fake_items(n_groups: int = 20, per_group: int = 4) -> tuple[list[dict], np.ndarray]:
    items, gid = [], []
    for g in range(n_groups):
        for k in range(per_group):
            items.append({"path": Path(f"/tmp/g{g}_{k}.jpg"), "label": "Free",
                          "exp": f"exp{k % 6 + 1}", "num": g * 100 + k,
                          "is_defect": g % 2})
            gid.append(g)
    return items, np.array(gid)


def test_grouped_split_never_straddles():
    items, gid = fake_items()
    for seed in range(20):
        tr, te = split_grouped(items, gid, np.random.default_rng(seed))
        assert set(gid[tr]).isdisjoint(set(gid[te])), f"seed {seed} 有分組跨越切分"
        assert set(tr) | set(te) == set(range(len(items)))   # 不漏樣本
        assert not (set(tr) & set(te))                        # 不重複


def test_grouped_split_reaches_target_size():
    """隔離做到了但測試集只剩 3 張，也是一種假的乾淨。"""
    items, gid = fake_items()
    tr, te = split_grouped(items, gid, np.random.default_rng(0))
    assert 0.15 * len(items) <= len(te) <= 0.40 * len(items)


def test_probe_random_split_does_straddle():
    """探針：隨機切分**必須**驗出跨越——驗不出來代表檢測本身沒作用。

    若這條測試變綠（隨機切分居然沒跨越），表示上面那條的通過毫無意義。
    """
    items, gid = fake_items()
    straddled = 0
    for seed in range(20):
        tr, te = split_random(items, gid, np.random.default_rng(seed))
        if set(gid[tr]) & set(gid[te]):
            straddled += 1
    assert straddled == 20, "隨機切分竟然沒有分組跨越，檢測邏輯有問題"


def test_group_ids_merges_identical_images(tmp_path):
    """完全相同的兩張影像必須併成同一組；毫不相關的不併。"""
    from PIL import Image

    rng = np.random.default_rng(0)
    a = (rng.random((64, 64)) * 255).astype("uint8")
    b = (rng.random((64, 64)) * 255).astype("uint8")
    paths = []
    for name, arr in (("a1", a), ("a2", a), ("b1", b)):
        p = tmp_path / f"{name}.jpg"
        Image.fromarray(arr).save(p, quality=95)
        paths.append(p)
    items = [{"path": p, "label": "Free", "exp": "exp1", "num": i, "is_defect": 0}
             for i, p in enumerate(paths)]
    gid = group_ids(items)
    assert gid[0] == gid[1], "同一張影像的兩份副本沒有併組——洩漏會殘留"
    assert gid[2] != gid[0], "不相關的影像被誤併——評估會過度保守"


@pytest.mark.parametrize("seed", range(5))
def test_no_empty_test_class(seed):
    """測試集必須同時含有瑕疵與良品，否則 F1／AUC 沒有意義。"""
    items, gid = fake_items()
    _, te = split_grouped(items, gid, np.random.default_rng(seed))
    labels = {items[i]["is_defect"] for i in te}
    assert labels == {0, 1}
