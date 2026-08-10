"""在封存的外層 test 上做正式評估——全專案唯一會呼叫 `unseal()` 的地方。

集中在一個檔案是刻意的：`grep -rn unseal src/` 就能列出「什麼時候讀了 test」，
而每次開封都必須附理由並落檔。搜尋（`search.py`）與架構的 lr 掃描都拿不到
這裡的東西。

這一層的雜訊帶與 README 第二節的**不是同一個東西**，不可互比：

- README 第二節：每顆種子換一次分組留出切分 → 雜訊帶含「切分變異＋訓練隨機性」
- 這裡：切分固定（`OUTER_SEED`），只換訓練種子 → 雜訊帶只含訓練隨機性，必然較窄

固定切分是為了讓不同設定之間是**成對比較**（同一批磁磚、同一批 test），
代價是這個 test 只有一批，換一批磁磚的浮動不在這條帶子裡。
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from experiment import Config, fit_eval
from nested import outer_split


def verdict(p: float) -> str:
    """與 analyze_results.py 同一組門檻，避免兩處各自漂移。"""
    from analyze_results import verdict as _v
    return _v(p)


def eval_on_sealed(items, gid, cfg: Config, seeds: range, device, reason: str,
                   outer_seed: int, task: str = "binary") -> tuple[list[dict], list[str]]:
    """開封一次，跑 N 顆訓練種子。回傳 (每顆種子的結果, 開封紀錄)。"""
    dev, sealed = outer_split(items, gid, outer_seed)
    te = sealed.unseal(reason)
    rows = []
    for seed in seeds:
        r = fit_eval(items, dev, te, cfg, seed, device, task)
        rows.append(r | {"config": cfg.to_dict()})
    return rows, sealed.opens


def summarise(rows: list[dict], metric: str = "f1") -> dict:
    xs = np.array([r[metric] for r in rows if r.get(metric) is not None])
    return {"mean": round(float(xs.mean()), 4), "sd": round(float(xs.std(ddof=1)), 4),
            "min": round(float(xs.min()), 4), "max": round(float(xs.max()), 4),
            "n": int(len(xs))}


def compare(a: list[dict], b: list[dict], metric: str = "f1") -> dict:
    """Welch 檢定（不假設等變異）＋寫死的判定字串。"""
    xa = np.array([r[metric] for r in a if r.get(metric) is not None])
    xb = np.array([r[metric] for r in b if r.get(metric) is not None])
    t, p = stats.ttest_ind(xa, xb, equal_var=False)
    return {"metric": metric, "a_mean": round(float(xa.mean()), 4),
            "b_mean": round(float(xb.mean()), 4),
            "gap": round(float(xa.mean() - xb.mean()), 4),
            "welch_t": round(float(t), 3), "p_value": round(float(p), 4),
            "verdict": verdict(float(p))}
