"""搜尋跑完之後的第二個問題：哪些超參維度真的有影響？

只知道「第 17 組勝出」用處不大——換一批資料就要重搜。有用的是
**哪幾個維度在這份資料上真的推得動指標**：下次搜尋可以把預算集中在那裡，
而被證明沒差別的維度可以直接鎖死成便宜的那個值。

作法是邊際平均（marginal mean）：把 32 組設定按某個維度的取值分堆，
比較各堆的 val F1 平均。隨機搜尋讓其他維度在每一堆裡都是隨機分布的，
所以堆與堆之間的差可以粗略讀成該維度的效果——**但只是粗略**：

- 每堆只有十來組，堆間差距小於同分帶時一律判「分不出來」，不是「沒有效果」
- 維度之間可能交互作用（例如 lr 的最佳值取決於 optimizer），邊際平均看不見
- 這是**觀察**不是實驗：要宣稱某個維度的因果效果，得固定其他維度單獨掃

所以輸出一律附上每堆的 n 與判定，判定沿用同一組寫死的門檻。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_results import verdict
from search import SPACE

RESULTS = Path(__file__).resolve().parent.parent / "results"


def marginals(rows: list[dict], band: float) -> dict:
    out = {}
    for dim in list(SPACE) + ["lr_decade"]:
        buckets: dict[str, list[float]] = {}
        for r in rows:
            cfg = r["config"]
            key = (f"1e{int(np.floor(np.log10(cfg['lr'])))}" if dim == "lr_decade"
                   else str(cfg[dim]))
            buckets.setdefault(key, []).append(r["val_f1_mean"])
        if len(buckets) < 2:
            continue
        stat = {k: {"mean": round(float(np.mean(v)), 4), "n": len(v)}
                for k, v in sorted(buckets.items(), key=lambda kv: -np.mean(kv[1]))}
        means = [s["mean"] for s in stat.values()]
        spread = round(max(means) - min(means), 4)
        # 兩堆時給 Welch p；多堆時只報跨度與同分帶的關係（多重比較不做校正就報 p
        # 會製造假訊號，寧可不報）。
        thin = min(s["n"] for s in stat.values())
        p = None
        if len(buckets) == 2 and thin >= 3:
            a, b = list(buckets.values())
            p = round(float(stats.ttest_ind(a, b, equal_var=False).pvalue), 4)
        if thin < 3:
            # n<3 的堆算不出可用的 p（Welch 會回 nan），而 nan 送進判定函式會
            # 靜默落到「落在雜訊帶內」——一個看起來很正常的結論，實際上沒有依據。
            v = f"各取值樣本不足（最小 n={thin}），不判定"
        elif p is not None and p < 0.05 and spread <= band:
            # 兩件事不同：p 小代表「兩堆平均確實不同」，同分帶代表「這個差距
            # 大到足以據此選設定」。堆間差異穩定但比一次重跑的浮動還小時，
            # 它是真的、但不值得為它換設定——這兩句話必須同時說。
            v = "統計上可分辨，但差距小於同分帶，不足以據此選設定"
        elif p is not None:
            v = verdict(p)
        else:
            v = "跨度大於同分帶，值得再查" if spread > band else "跨度小於同分帶，分不出來"
        out[dim] = {"buckets": stat, "spread": spread, "p_value": p,
                    "min_bucket_n": thin, "verdict": v}
    return out


def decompose(rows: list[dict]) -> dict:
    """把 設定 × 折 的成績拆成「折主效應／設定主效應／殘差」。

    同分帶寬到 ±0.117 時，第一個要問的是**這個浮動是誰造成的**：

    - 若主要來自**折**（某一折對所有設定都難），那是共同的加法偏移，
      成對比較（同一批折上比兩組）就能消掉一部分——帶不該用它撐開。
    - 若主要來自**殘差**（這組設定剛好在這折特別好），那才是真雜訊，
      只能靠增加折數壓下去。

    這裡順便量出成對後還剩多少：兩兩相減後的逐折差 sd 中位數。
    """
    M = np.array([[f["f1"] for f in sorted(r["folds"], key=lambda x: x["fold"])]
                  for r in rows])
    n_cfg, n_fold = M.shape
    grand = M.mean()
    fold_mean, cfg_mean = M.mean(0), M.mean(1)
    ss_total = float(((M - grand) ** 2).sum())
    ss_fold = float(n_cfg * ((fold_mean - grand) ** 2).sum())
    ss_cfg = float(n_fold * ((cfg_mean - grand) ** 2).sum())
    pair_sds = [float((M[i] - M[j]).std(ddof=1))
                for i in range(n_cfg) for j in range(i + 1, n_cfg)]
    sds = np.array([r["val_f1_sd"] for r in rows])
    means = np.array([r["val_f1_mean"] for r in rows])
    pooled = float(np.sqrt((sds ** 2).mean()))
    return {
        "fold_means": [round(float(x), 4) for x in fold_mean],
        "share_fold_effect": round(ss_fold / ss_total, 4),
        "share_config_effect": round(ss_cfg / ss_total, 4),
        "share_residual": round((ss_total - ss_fold - ss_cfg) / ss_total, 4),
        "pooled_sd": round(pooled, 4),
        "paired_diff_sd_median": round(float(np.median(pair_sds)), 4),
        # 折間完全沒有浮動時 pooled 是 0，相除會炸；那種情況本來就無從談「消掉幾成」。
        "variance_removed_by_pairing": (
            None if pooled == 0
            else round(1 - float(np.median(pair_sds)) / pooled, 4)),
        # 爛設定會不會把合併 sd 拉大？相關係數接近 0 就不是這個原因。
        # 任一邊零變異時相關係數無定義（np 會回 nan），明寫 None 不留 nan。
        "corr_mean_sd": (None if means.std() == 0 or sds.std() == 0
                         else round(float(np.corrcoef(means, sds)[0, 1]), 4)),
        "n_configs": n_cfg, "n_folds": n_fold,
    }


def folds_needed(paired_sd: float, gap: float, mult: float = 2.0) -> int:
    """要讓 `gap` 這麼大的差距落到同分帶外，需要幾折。

    帶寬＝mult × paired_sd / √k，要落到帶外得 **嚴格** 小於 gap，
    因為同分帶用的是 `>= best − band`（剛好等於帶寬的仍算同分）。
    於是 k > (mult·sd/gap)²，恰好整除時要再加一折。

    這個數字回答的是「下一輪該加到幾折」，不是回頭改這一輪的判定。
    """
    x = (mult * paired_sd / gap) ** 2
    return int(np.floor(x)) + 1 if abs(x - round(x)) < 1e-9 else int(np.ceil(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default="search_results.json")
    ap.add_argument("--out", default="search_analysis.json")
    a = ap.parse_args()

    d = json.loads((RESULTS / a.search).read_text(encoding="utf-8"))
    rows, sel = d["rows"], d["selection"]
    band = sel["tie_band"]
    f1s = [r["val_f1_mean"] for r in rows]
    print(f"{len(rows)} 組設定｜val F1 {min(f1s):.4f}–{max(f1s):.4f}｜同分帶 ±{band:.4f}"
          f"｜帶內 {len(sel['tie_set'])} 組")

    dec = decompose(rows)
    top, win = rows[sel["best_by_mean"]], rows[sel["winner"]]
    gap = round(top["val_f1_mean"] - win["val_f1_mean"], 4)
    need = folds_needed(dec["paired_diff_sd_median"], gap) if gap > 0 else None
    print(f"\n變異拆解：折主效應 {dec['share_fold_effect']:.1%}"
          f"｜設定主效應 {dec['share_config_effect']:.1%}"
          f"｜殘差 {dec['share_residual']:.1%}｜各折平均 {dec['fold_means']}")
    print(f"  合併 sd {dec['pooled_sd']} → 成對後 {dec['paired_diff_sd_median']}"
          f"（消掉 {dec['variance_removed_by_pairing']:.0%}）"
          f"｜mean-sd 相關 {dec['corr_mean_sd']}")
    if need:
        print(f"  帶內第一名與勝出者差 {gap} → 要讓它落到帶外需 {need} 折"
              f"（本次 {dec['n_folds']} 折）")

    m = marginals(rows, band)
    print(f"\n{'維度':16s}{'跨度':>8s}  各取值（平均 F1／n）")
    for dim, v in sorted(m.items(), key=lambda kv: -kv[1]["spread"]):
        cells = "、".join(f"{k}={s['mean']:.3f}({s['n']})" for k, s in v["buckets"].items())
        print(f"{dim:16s}{v['spread']:>8.4f}  {cells}")
        print(f"{'':16s}{'':>8s}  → {v['verdict']}"
              + (f"（p={v['p_value']}）" if v["p_value"] is not None else ""))

    out = {"n_configs": len(rows), "tie_band": band, "tie_set_size": len(sel["tie_set"]),
           "val_f1_range": [round(min(f1s), 4), round(max(f1s), 4)],
           "winner": sel["winner"], "decomposition": dec,
           "gap_top_vs_winner": gap, "folds_needed_for_that_gap": need,
           "marginals": m,
           "caveat": "邊際平均是觀察不是實驗：堆內其他維度隨機、堆間可能有交互作用，"
                     "跨度小於同分帶只代表分不出來，不代表沒有效果。"}
    (RESULTS / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                 encoding="utf-8")


if __name__ == "__main__":
    main()
