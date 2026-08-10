"""超參數隨機搜尋：只看內層 validation，外層 test 從頭到尾封存。

**這份搜尋的重點不是找到更高的分數，是讓「更高的分數」有機會是真的。**
在洩漏未修正的切分上搜尋，搜到的是「最會背磁磚的那組超參」；本檔跑在
`nested.py` 的三層結構上，外層 test 連傳都沒傳進來。

兩個刻意寫死在程式裡、跑之前就定案的規則（避免看到結果再挑標準）：

1. **同分帶（tie band）**：三折之間的成績本來就會浮動。best 之外，凡是
   落在 `best_mean − 2 × 折間標準誤` 內的設定，統計上與 best 沒有差別，
   一律視為同分。
2. **同分就挑最便宜的**：同分帶內以「訓練時間中位數」最小者勝出，再同分
   則取 epochs 少者、再同分取先抽到者。理由是這個帶內的名次差異是雜訊，
   拿雜訊當理由去選一個更貴的模型，代價卻是真的。

為什麼用隨機搜尋而不是網格：同樣的預算下，網格會把大量嘗試花在對結果
不敏感的維度上（例如 batch size），隨機搜尋在每個維度都取到不同的值。
（Bergstra & Bengio 2012）

而 lr 的抽樣範圍依 optimizer 而定——AdamW 與 SGD 的合理區間差兩個數量級，
共用一個區間等於系統性地讓其中一方跑在錯的量級上，那不是公平比較。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from experiment import Config, fit_eval, pick_device, preload
from nested import inner_folds, outer_split
from tiles import group_ids, list_images

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUTER_SEED = 100          # 外層切分固定：所有 v2 數字共用同一批封存 test
FOLD_SEED = 7
TRAIN_SEED = 0            # 搜尋期間訓練種子固定，讓設定之間是成對比較
TIE_BAND_SD_MULT = 2.0    # 同分帶寬度＝這個倍數 × 折間標準誤
PATIENCE = 3
SELECTION_RULE = (
    "1) 以三折 val F1 平均排序；"
    "2) 同分帶＝best_mean − 2.0 × 折間標準誤，帶內視為統計上無差別；"
    "3) 同分帶內取訓練時間中位數最小者，再同分取 epochs 少者、再同分取先抽到者。"
)

SPACE = {
    "arch": ["resnet18"],          # 架構對照另跑（arch_compare.py），這裡固定
    "epochs": [4, 6, 8],
    "freeze": ["none", "backbone", "partial"],
    "scheduler": ["none", "cosine", "onecycle"],
    "optimizer": ["adamw", "sgd"],
    "aug": ["flips", "flips_rot_jitter", "randaugment"],
    "label_smoothing": [0.0, 0.1],
    "img_size": [224, 160],
    "batch": [32, 64],
    "weight_decay": [0.0, 1e-4, 1e-2],
}
LR_RANGE = {"adamw": (5e-5, 3e-3), "sgd": (1e-3, 1e-1)}


def sample_config(rng: np.random.Generator) -> Config:
    pick = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
    lo, hi = LR_RANGE[pick["optimizer"]]
    lr = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return Config(lr=round(lr, 6), **pick)


def select(rows: list[dict]) -> dict:
    """套用 SELECTION_RULE。輸入是每個設定的彙整列，輸出含同分帶與贏家。"""
    ranked = sorted(rows, key=lambda r: -r["val_f1_mean"])
    best = ranked[0]
    # 折間標準誤：用所有設定的折內標準差合併，單一設定的三折 sd 本身太不穩。
    pooled_sd = float(np.sqrt(np.mean([r["val_f1_sd"] ** 2 for r in rows])))
    band = TIE_BAND_SD_MULT * pooled_sd / np.sqrt(best["n_folds"])
    tie = [r for r in ranked if r["val_f1_mean"] >= best["val_f1_mean"] - band]
    winner = min(tie, key=lambda r: (r["train_sec_median"], r["config"]["epochs"], r["idx"]))
    return {"pooled_fold_sd": round(pooled_sd, 4), "tie_band": round(float(band), 4),
            "best_by_mean": best["idx"], "tie_set": [r["idx"] for r in tie],
            "winner": winner["idx"], "rule": SELECTION_RULE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=32)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--out", default="search_results.json")
    a = ap.parse_args()

    device = pick_device()
    items = list_images()
    gid = group_ids(items)
    dev, sealed = outer_split(items, gid, OUTER_SEED)
    folds = inner_folds(dev, gid, FOLD_SEED, n_folds=a.folds)
    for size in set(SPACE["img_size"]):
        preload(items, size)
    print(f"{len(items)} 張／{gid.max() + 1} 組｜dev {len(dev)}｜封存 test {len(sealed)}｜"
          f"{a.folds} 折 val {[len(v) for _, v in folds]}｜裝置 {device}")

    rng = np.random.default_rng(a.sample_seed)
    rows, out_path = [], RESULTS / a.out
    RESULTS.mkdir(exist_ok=True)
    for i in range(a.configs):
        cfg = sample_config(rng)
        runs = []
        for k, (tr, va) in enumerate(folds):
            r = fit_eval(items, tr, va, cfg, TRAIN_SEED, device, "binary",
                         eval_each_epoch=True, early_stop="f1", patience=PATIENCE)
            runs.append({"fold": k, "best_epoch": r["best_epoch"],
                         "epochs_run": r["epochs_run"], "train_sec": r["train_sec"],
                         **{m: r["best_epoch_metrics"][m] for m in ("f1", "auc", "accuracy")}})
        f1s = [x["f1"] for x in runs]
        rows.append({
            "idx": i, "config": cfg.to_dict(),
            "val_f1_mean": round(float(np.mean(f1s)), 4),
            "val_f1_sd": round(float(np.std(f1s, ddof=1)), 4),
            "val_auc_mean": round(float(np.mean([x["auc"] for x in runs])), 4),
            "best_epoch_median": int(statistics.median_low([x["best_epoch"] for x in runs])),
            "train_sec_median": float(statistics.median([x["train_sec"] for x in runs])),
            "n_folds": len(runs), "folds": runs,
        })
        print(f"  [{i + 1:>2}/{a.configs}] val F1 {rows[-1]['val_f1_mean']:.4f}"
              f" ±{rows[-1]['val_f1_sd']:.4f}｜{rows[-1]['train_sec_median']:.0f}s"
              f"｜{cfg.arch} lr={cfg.lr:g} {cfg.optimizer} {cfg.freeze}"
              f" {cfg.scheduler} {cfg.aug} ls={cfg.label_smoothing}"
              f" sz={cfg.img_size} bs={cfg.batch} wd={cfg.weight_decay:g}"
              f"｜best_ep={rows[-1]['best_epoch_median']}")
        # 每個設定跑完就落檔：4 顆小時的搜尋中途斷掉時，已跑的不用重跑。
        out_path.write_text(json.dumps(
            {"outer_seed": OUTER_SEED, "fold_seed": FOLD_SEED, "train_seed": TRAIN_SEED,
             "n_dev": len(dev), "n_sealed_test": len(sealed), "patience": PATIENCE,
             "space": SPACE, "lr_range": LR_RANGE, "rows": rows,
             "selection": select(rows), "sealed_opens": sealed.opens},
            ensure_ascii=False, indent=1), encoding="utf-8")

    sel = select(rows)
    win = rows[sel["winner"]]
    assert not sealed.opens, "搜尋期間讀了封存 test——結果作廢"
    print(f"\n同分帶 ±{sel['tie_band']:.4f}（折間 sd {sel['pooled_fold_sd']:.4f}）｜"
          f"帶內 {len(sel['tie_set'])} 組｜平均最高是 #{sel['best_by_mean']}"
          f"（{rows[sel['best_by_mean']]['val_f1_mean']:.4f}）")
    print(f"依規則勝出：#{sel['winner']} val F1 {win['val_f1_mean']:.4f}、"
          f"{win['train_sec_median']:.0f}s、best_epoch {win['best_epoch_median']}")
    print(json.dumps(win["config"], ensure_ascii=False))


if __name__ == "__main__":
    main()
