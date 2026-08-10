"""架構對照：四個 backbone 在同一份封存 test 上比，各自調過 lr 才比。

**沿用「為 ResNet-18 調好的超參」去比別的架構，比出來的是超參的主場優勢。**
所以每個架構先在內層 val 上掃自己的 lr（其餘超參沿用搜尋勝出的那組），
再用各自最好的 lr 進正式評估。這一步不便宜，但少了它，結論只是
「別人穿我的鞋跑比較慢」。

另一半同樣重要：**比較的結論要對照雜訊帶讀**。四個架構的差距若都落在
種子間浮動之內，正確的結論是「準確率上分不出高下」——這種時候決策軸就換成
延遲與模型大小（見 `latency.py`），而不是硬挑一個平均值最高的宣稱它最好。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment import ARCHS, Config, fit_eval, pick_device, preload
from final_eval import load_winner
from nested import inner_folds, outer_split
from search import FOLD_SEED, OUTER_SEED, TRAIN_SEED
from sealed_eval import compare, eval_on_sealed, summarise
from tiles import group_ids, list_images

RESULTS = Path(__file__).resolve().parent.parent / "results"
# lr 掃描格點依 optimizer 而定：AdamW 與 SGD 的合理區間差兩個數量級。
LR_GRID = {"adamw": [1e-4, 3e-4, 1e-3], "sgd": [3e-3, 1e-2, 3e-2]}
PATIENCE = 3


def tune_lr(items, gid, folds, base_cfg: Config, arch: str, device) -> dict:
    """在內層 val 上為單一架構掃 lr。回傳最佳 lr 與每個格點的成績。"""
    rows = []
    for lr in LR_GRID[base_cfg.optimizer]:
        cfg = Config(**(base_cfg.to_dict() | {"arch": arch, "lr": lr}))
        f1s, eps = [], []
        for tr, va in folds:
            r = fit_eval(items, tr, va, cfg, TRAIN_SEED, device, "binary",
                         eval_each_epoch=True, early_stop="f1", patience=PATIENCE)
            f1s.append(r["best_epoch_metrics"]["f1"])
            eps.append(r["best_epoch"])
        rows.append({"lr": lr, "val_f1_mean": round(float(np.mean(f1s)), 4),
                     "val_f1_sd": round(float(np.std(f1s, ddof=1)), 4),
                     "best_epoch_median": int(np.median(eps)), "folds_f1": f1s})
        print(f"    lr={lr:<8g} val F1 {rows[-1]['val_f1_mean']:.4f}"
              f" ±{rows[-1]['val_f1_sd']:.4f}｜best_ep={rows[-1]['best_epoch_median']}")
    best = max(rows, key=lambda r: r["val_f1_mean"])
    return {"grid": rows, "best_lr": best["lr"], "epochs": best["best_epoch_median"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--search", default="search_results.json")
    ap.add_argument("--out", default="arch_results.json")
    a = ap.parse_args()

    base_cfg, _ = load_winner(RESULTS / a.search)
    device = pick_device()
    items = list_images()
    gid = group_ids(items)
    preload(items, base_cfg.img_size)
    dev, sealed = outer_split(items, gid, OUTER_SEED)
    folds = inner_folds(dev, gid, FOLD_SEED)
    print(f"共用超參（lr 除外）：{json.dumps(base_cfg.to_dict(), ensure_ascii=False)}")

    out = {"outer_seed": OUTER_SEED, "seeds": a.seeds, "lr_grid": LR_GRID,
           "shared_config": base_cfg.to_dict(), "archs": {}, "sealed_opens": []}
    out_path, runs_by_arch = RESULTS / a.out, {}
    for arch in a.archs:
        print(f"\n{arch}：內層 val 掃 lr")
        tuned = tune_lr(items, gid, folds, base_cfg, arch, device)
        cfg = Config(**(base_cfg.to_dict() | {"arch": arch, "lr": tuned["best_lr"],
                                              "epochs": tuned["epochs"]}))
        rows, opens = eval_on_sealed(items, gid, cfg, range(a.seeds), device,
                                     f"架構對照正式評估：{arch}", OUTER_SEED)
        runs_by_arch[arch] = rows
        out["archs"][arch] = {
            "config": cfg.to_dict(), "lr_tuning": tuned,
            "sealed": {m: summarise(rows, m) for m in ("f1", "auc", "accuracy")},
            "train_sec_median": float(np.median([r["train_sec"] for r in rows])),
            "runs": rows,
        }
        out["sealed_opens"] += opens
        s = out["archs"][arch]["sealed"]["f1"]
        print(f"  → 封存 test F1 {s['mean']:.4f} ±{s['sd']:.4f}"
              f"（lr={tuned['best_lr']:g}, epochs={tuned['epochs']}）")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    ref = a.archs[0]
    out["vs_" + ref] = {arch: {m: compare(runs_by_arch[arch], runs_by_arch[ref], m)
                               for m in ("f1", "auc", "accuracy")}
                        for arch in a.archs if arch != ref}
    # 全體雜訊帶：各架構種子間 sd 的合併值。四個架構的差距若都小於它，
    # 「哪個架構比較準」這個問題在這份資料上沒有答案。
    out["pooled_seed_sd_f1"] = round(float(np.sqrt(np.mean(
        [out["archs"][x]["sealed"]["f1"]["sd"] ** 2 for x in a.archs]))), 4)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n種子間合併 sd（F1）：{out['pooled_seed_sd_f1']}")
    for arch, cmp_ in out["vs_" + ref].items():
        c = cmp_["f1"]
        print(f"  {arch:20s} vs {ref}: {c['gap']:+.4f}（p={c['p_value']}）→ {c['verdict']}")


if __name__ == "__main__":
    main()
