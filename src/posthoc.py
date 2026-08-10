"""事後診斷：搜尋失手時，要分得清是「規則挑錯」還是「搜尋本身沒訊號」。

正式評估顯示搜尋勝出的設定在封存 test 上輸給 BASELINE。有兩種可能，
處方完全相反：

A. **規則挑錯**——同分帶內取最便宜的那一步選錯了，val 平均最高的那組其實更好。
   若是這樣，該檢討的是挑選規則。
B. **搜尋沒訊號**——三折 val 的浮動大到名次本身就是雜訊，挑誰都一樣。
   若是這樣，該檢討的是折數與資料量，換挑選規則沒有用。

兩個診斷各自對應一種可能：

1. 把 val 平均最高的那組也放到封存 test 上（分辨 A）
2. 把 BASELINE 放回同一組內層 val，看它會排第幾（分辨 B）——
   搜尋若有訊號，一個在 test 上贏的設定不該在 val 上排很後面

**這兩個數字是看過正式評估結果之後才追加的，屬事後分析。** 它們用來解釋
已經發生的事，不能回頭當成挑設定的依據——那會變成拿 test 挑超參，
正是整個巢狀結構要擋的事。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment import BASELINE, Config, fit_eval, pick_device, preload
from nested import inner_folds, outer_split
from search import FOLD_SEED, OUTER_SEED, PATIENCE, TRAIN_SEED
from sealed_eval import compare, eval_on_sealed, summarise
from tiles import group_ids, list_images

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--search", default="search_results.json")
    ap.add_argument("--final", default="final_eval.json")
    ap.add_argument("--out", default="posthoc.json")
    a = ap.parse_args()

    search = json.loads((RESULTS / a.search).read_text(encoding="utf-8"))
    final = json.loads((RESULTS / a.final).read_text(encoding="utf-8"))
    rows, sel = search["rows"], search["selection"]
    device = pick_device()
    items = list_images()
    gid = group_ids(items)
    preload(items, 224)
    preload(items, 160)

    # 診斷 1：val 平均最高的那組，在封存 test 上表現如何
    top = rows[sel["best_by_mean"]]
    cfg_top = Config(**(top["config"] | {"epochs": top["best_epoch_median"]}))
    top_runs, opens = eval_on_sealed(items, gid, cfg_top, range(a.seeds), device,
                                     "事後診斷：val 平均最高的設定（非選擇依據）",
                                     OUTER_SEED)
    base_runs = final["runs"]["baseline"]
    print(f"#{sel['best_by_mean']}（val 最高 {top['val_f1_mean']:.4f}）"
          f"→ 封存 test F1 {summarise(top_runs)['mean']:.4f}")

    # 診斷 2：BASELINE 放回內層 val，看它排第幾
    dev, _ = outer_split(items, gid, OUTER_SEED)
    folds = inner_folds(dev, gid, FOLD_SEED)
    f1s = []
    for tr, va in folds:
        r = fit_eval(items, tr, va, BASELINE, TRAIN_SEED, device, "binary",
                     eval_each_epoch=True, early_stop="f1", patience=PATIENCE)
        f1s.append(r["best_epoch_metrics"]["f1"])
    base_val = float(np.mean(f1s))
    rank = 1 + sum(1 for r in rows if r["val_f1_mean"] > base_val)
    print(f"BASELINE 在同一組 val：F1 {base_val:.4f} → 32 組中排第 {rank}")

    out = {
        "note": "本檔數字為事後診斷，看過正式評估後才追加，不得回頭當作挑選依據。",
        "seeds": a.seeds, "outer_seed": OUTER_SEED,
        "top_by_val": {
            "idx": sel["best_by_mean"], "config": cfg_top.to_dict(),
            "search_val_f1": top["val_f1_mean"],
            "sealed": {m: summarise(top_runs, m) for m in ("f1", "auc", "accuracy")},
            "vs_baseline": {m: compare(top_runs, base_runs, m)
                            for m in ("f1", "auc", "accuracy")},
            "selection_optimism_f1": round(
                top["val_f1_mean"] - summarise(top_runs)["mean"], 4),
        },
        "baseline_on_inner_val": {
            "val_f1_mean": round(base_val, 4),
            "val_f1_sd": round(float(np.std(f1s, ddof=1)), 4),
            "folds_f1": [round(x, 4) for x in f1s],
            "rank_among_searched": rank, "n_searched": len(rows),
            "tie_band": sel["tie_band"],
            "inside_tie_band": bool(base_val >= rows[sel["best_by_mean"]]["val_f1_mean"]
                                    - sel["tie_band"]),
        },
        "sealed_opens": opens,
        "runs": {"top_by_val": top_runs},
    }
    (RESULTS / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    c = out["top_by_val"]["vs_baseline"]["f1"]
    print(f"#{sel['best_by_mean']} vs BASELINE：{c['gap']:+.4f}（p={c['p_value']}）"
          f"→ {c['verdict']}")


if __name__ == "__main__":
    main()
