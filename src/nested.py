"""巢狀分組切分：搜尋只能看內層 validation，外層 test 封存到最後一刻。

超參搜尋比訓練更容易洩漏，而且洩漏得更隱蔽：切分寫對了、模型也沒作弊，
但**只要拿 test 的成績來挑超參，挑出來的那組就已經看過 test 了**——回報的
數字會同時包含「這組超參真的比較好」和「這組超參剛好合這批 test 的胃口」，
而後者不會在下一批磁磚上重現。三層結構因此是必要的，不是講究：

    全部 1,344 張
    ├── 外層 test（封存；整組留出、搜尋期間不可讀）        ← 最後只讀一次
    └── dev
        ├── 內層 train ┐ 三折，每折的 val 是不同的磁磚分組
        └── 內層 val   ┘ 超參的所有決定（含 early stopping）都在這裡做

而且**三層的每一刀都必須是分組留出**。內層若用隨機切分，val 上的成績同樣
被近乎重複影像灌水，據此挑出來的超參是「最會背磁磚的那組」——外層切得再乾淨
也救不回來，因為錯誤發生在挑選階段。

封存靠 `SealedTest`：它拿不出索引，除非呼叫 `unseal(reason)`，而每次 unseal
都會留下紀錄落檔。搜尋程式從頭到尾拿不到能迭代的東西，想偷看必須寫一行
`unseal(...)`——那行在 diff 裡是看得見的。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TEST_FRAC = 0.25
VAL_FRAC = 0.25          # 佔 dev 的比例
N_FOLDS = 3


class SealedError(RuntimeError):
    """有人在封存期間想讀 test。"""


@dataclass
class SealedTest:
    """封存的外層測試集。知道它有多大是允許的，讀到它是誰不行。"""

    _idx: tuple[int, ...]
    label: str = "outer-test"
    opens: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._idx)

    def __iter__(self):
        raise SealedError(f"{self.label} 已封存：要讀請明寫 unseal(reason)")

    def __getitem__(self, i):
        raise SealedError(f"{self.label} 已封存：要讀請明寫 unseal(reason)")

    def unseal(self, reason: str) -> list[int]:
        """開封並留痕。reason 會被寫進結果檔，讓「什麼時候讀了 test」可稽核。"""
        if not reason or not reason.strip():
            raise SealedError("開封必須說明理由")
        self.opens.append(reason)
        return list(self._idx)


def _groups_until(gid: np.ndarray, pool: list[int], order: np.ndarray,
                  target: int) -> set[int]:
    """依 order 收分組，直到覆蓋的樣本數達到 target。"""
    in_pool = np.zeros(int(gid.max()) + 1, dtype=bool)
    counts = np.zeros(int(gid.max()) + 1, dtype=int)
    for i in pool:
        in_pool[gid[i]] = True
        counts[gid[i]] += 1
    picked, n = set(), 0
    for g in order:
        if n >= target:
            break
        g = int(g)
        if not in_pool[g]:
            continue
        picked.add(g)
        n += counts[g]
    return picked


def outer_split(items: list[dict], gid: np.ndarray, seed: int,
                test_frac: float = TEST_FRAC) -> tuple[list[int], SealedTest]:
    """整組留出外層 test 並直接封存。回傳 (dev 索引, SealedTest)。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(int(gid.max()) + 1)
    pool = list(range(len(items)))
    test_groups = _groups_until(gid, pool, order, int(len(items) * test_frac))
    dev = [i for i in pool if gid[i] not in test_groups]
    test = [i for i in pool if gid[i] in test_groups]
    return dev, SealedTest(tuple(test), label=f"outer-test(seed={seed})")


def inner_folds(dev: list[int], gid: np.ndarray, seed: int, n_folds: int = N_FOLDS,
                val_frac: float = VAL_FRAC) -> list[tuple[list[int], list[int]]]:
    """在 dev 內切 n_folds 組分組留出的 (train, val)。

    每折換一次分組排列，所以不同折的 val 是不同的磁磚；同一折內
    val 的分組絕不出現在 train。刻意不用 sklearn 的 GroupKFold：
    那會讓每折 val 大小由分組數決定，這裡要的是固定比例。
    """
    out = []
    for k in range(n_folds):
        rng = np.random.default_rng(seed * 1000 + k)
        order = rng.permutation(int(gid.max()) + 1)
        val_groups = _groups_until(gid, dev, order, int(len(dev) * val_frac))
        tr = [i for i in dev if gid[i] not in val_groups]
        va = [i for i in dev if gid[i] in val_groups]
        out.append((tr, va))
    return out


def naive_inner_folds(dev: list[int], gid: np.ndarray, seed: int,
                      n_folds: int = N_FOLDS,
                      val_frac: float = VAL_FRAC) -> list[tuple[list[int], list[int]]]:
    """探針用的**壞**切分：內層改成隨機抽樣，分組會跨越 train／val。

    正式流程不呼叫它。它存在的唯一理由是讓
    `tests/test_nested_split.py` 能證明「內層洩漏偵測」真的驗得出東西——
    偵測器對著永遠乾淨的輸入永遠是綠的，那個綠沒有意義。
    """
    out = []
    for k in range(n_folds):
        rng = np.random.default_rng(seed * 1000 + k)
        perm = rng.permutation(dev)
        cut = int(len(dev) * val_frac)
        out.append((perm[cut:].tolist(), perm[:cut].tolist()))
    return out
