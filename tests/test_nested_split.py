"""巢狀切分的完整性：搜尋階段能不能看到 test，是超參結果可不可信的分水嶺。

外層切乾淨、內層卻隨機切，是這個階段最容易犯又最看不出來的錯：超參是在
被灌水的 val 上挑的，挑出來的是「最會背磁磚的那組」，而外層 test 只會忠實
回報一個比預期低的數字，不會告訴你原因。所以這裡釘三件事：

1. 內層 val 的分組不出現在內層 train（三層都必須是分組留出）
2. 封存的 test 分組不出現在任何一折的 train 或 val
3. `SealedTest` 真的擋得住讀取，而且開封留痕

每一條都配探針：對著永遠乾淨的輸入，偵測器永遠是綠的，那個綠沒有意義。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nested import (SealedError, SealedTest, inner_folds,  # noqa: E402
                    naive_inner_folds, outer_split)


def fake_items(n_groups: int = 40, per_group: int = 4):
    items, gid = [], []
    for g in range(n_groups):
        for k in range(per_group):
            items.append({"path": Path(f"/tmp/g{g}_{k}.jpg"), "label": "Free",
                          "exp": f"exp{k % 6 + 1}", "num": g * 100 + k,
                          "is_defect": g % 2})
            gid.append(g)
    return items, np.array(gid)


@pytest.mark.parametrize("seed", range(20))
def test_inner_folds_never_straddle(seed: int) -> None:
    items, gid = fake_items()
    dev, _ = outer_split(items, gid, seed)
    for k, (tr, va) in enumerate(inner_folds(dev, gid, seed)):
        assert set(gid[tr]).isdisjoint(set(gid[va])), f"seed {seed} 折 {k} 有分組跨越"
        assert set(tr) | set(va) == set(dev)
        assert not (set(tr) & set(va))


def test_probe_naive_inner_folds_do_straddle() -> None:
    """探針：內層改成隨機切分**必須**被驗出跨越。

    驗不出來，代表上面那條的通過只是因為檢測本身沒作用。
    """
    items, gid = fake_items()
    dev, _ = outer_split(items, gid, 0)
    straddled = sum(1 for tr, va in naive_inner_folds(dev, gid, 0)
                    if set(gid[tr]) & set(gid[va]))
    assert straddled == 3, "隨機內層切分竟然沒有分組跨越，檢測邏輯有問題"


@pytest.mark.parametrize("seed", range(20))
def test_sealed_test_groups_appear_in_no_fold(seed: int) -> None:
    """封存的 test 分組不得出現在任何一折——這是搜尋不洩漏的定義。"""
    items, gid = fake_items()
    dev, sealed = outer_split(items, gid, seed)
    test_groups = set(gid[sealed.unseal("測試用：驗證隔離")])
    for tr, va in inner_folds(dev, gid, seed):
        assert test_groups.isdisjoint(set(gid[tr]))
        assert test_groups.isdisjoint(set(gid[va]))


def test_folds_are_actually_different() -> None:
    """三折若切出同一批 val，等於只跑了一折——平均值會假裝比較穩。"""
    items, gid = fake_items()
    dev, _ = outer_split(items, gid, 0)
    vals = [frozenset(va) for _, va in inner_folds(dev, gid, 0)]
    assert len(set(vals)) == 3, "不同折的 validation 完全相同"


def test_split_sizes_are_sane() -> None:
    items, gid = fake_items()
    dev, sealed = outer_split(items, gid, 0)
    assert 0.15 * len(items) <= len(sealed) <= 0.40 * len(items)
    for _, va in inner_folds(dev, gid, 0):
        assert 0.15 * len(dev) <= len(va) <= 0.40 * len(dev)


def test_sealed_test_refuses_to_be_read() -> None:
    s = SealedTest((1, 2, 3))
    with pytest.raises(SealedError):
        list(s)
    with pytest.raises(SealedError):
        s[0]
    with pytest.raises(SealedError):
        s.unseal("")          # 開封必須說明理由
    assert len(s) == 3        # 知道它有多大是允許的
    assert s.opens == []


def test_unseal_leaves_a_trace() -> None:
    """開封留痕才能事後稽核「什麼時候讀了 test」。"""
    s = SealedTest((1, 2, 3))
    assert s.unseal("正式評估") == [1, 2, 3]
    assert s.opens == ["正式評估"]


def test_probe_a_peeking_search_would_be_caught() -> None:
    """探針：模擬一支「順手拿 test 當 val」的搜尋程式，必須當場炸掉。

    若這裡沒有例外，`SealedTest` 就只是個註解，擋不住任何事。
    """
    items, gid = fake_items()
    _, sealed = outer_split(items, gid, 0)

    def peeking_search(sealed_test):
        return [items[i] for i in sealed_test]   # 沒有 unseal 就想迭代

    with pytest.raises(SealedError):
        peeking_search(sealed)


def test_search_module_does_not_unseal() -> None:
    """搜尋程式碼裡不該出現 unseal——出現了就是在 diff 裡看得見的違規。"""
    src = (Path(__file__).resolve().parent.parent / "src" / "search.py").read_text(
        encoding="utf-8")
    assert "unseal" not in src, "search.py 出現 unseal：超參是看著 test 挑的，結果作廢"
