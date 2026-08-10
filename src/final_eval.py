"""正式評估：把搜尋選出的設定與 BASELINE 放到封存的 test 上，各跑 N 顆種子。

這一步要回答兩個問題，第二個比第一個重要：

1. 搜尋選出來的設定，在沒看過的磁磚上真的比較好嗎？
2. **搜尋期間那個 val 分數，高估了多少？**（selection optimism）

第二個問題是這整套巢狀結構的產物。搜尋在 32 組設定裡挑最高分，這個「挑最高」
本身就會挑到運氣好的那組——同一份 val 上，最大值是有偏的。val 與 test 的差
就是這份偏誤的量級，也就是「直接拿 test 挑超參」會灌進最終數字的水。

比較是成對的：兩組設定跑同一批封存 test、同一組訓練種子，差別只有超參。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment import BASELINE, Config, pick_device, preload
from search import OUTER_SEED
from sealed_eval import compare, eval_on_sealed, summarise
from tiles import group_ids, list_images

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_winner(path: Path) -> tuple[Config, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    sel = d["selection"]
    row = d["rows"][sel["winner"]]
    cfg = Config(**row["config"])
    # 搜尋階段的 early stopping 挑出的 epoch 數就是正式評估要跑的輪數，
    # 否則等於用 val 挑了 epoch、卻在 test 上跑另一個輪數。
    return Config(**(cfg.to_dict() | {"epochs": row["best_epoch_median"]})), row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--search", default="search_results.json")
    ap.add_argument("--out", default="final_eval.json")
    a = ap.parse_args()

    cfg, row = load_winner(RESULTS / a.search)
    device = pick_device()
    items = list_images()
    gid = group_ids(items)
    for size in {cfg.img_size, BASELINE.img_size}:
        preload(items, size)
    print(f"勝出設定：{json.dumps(cfg.to_dict(), ensure_ascii=False)}")
    print(f"搜尋期間 val F1 {row['val_f1_mean']:.4f}（三折）→ 現在到封存 test 上驗")

    seeds = range(a.seeds)
    tuned, opens_a = eval_on_sealed(items, gid, cfg, seeds, device,
                                    "正式評估：搜尋勝出設定", OUTER_SEED)
    print(f"  tuned    F1 {summarise(tuned)['mean']:.4f}")
    base, opens_b = eval_on_sealed(items, gid, BASELINE, seeds, device,
                                   "正式評估：BASELINE 對照", OUTER_SEED)
    print(f"  baseline F1 {summarise(base)['mean']:.4f}")

    out = {
        "outer_seed": OUTER_SEED, "seeds": a.seeds,
        "winner_config": cfg.to_dict(), "baseline_config": BASELINE.to_dict(),
        "search_val_f1": row["val_f1_mean"],
        "tuned": {m: summarise(tuned, m) for m in ("f1", "auc", "accuracy")},
        "baseline": {m: summarise(base, m) for m in ("f1", "auc", "accuracy")},
        "comparison": {m: compare(tuned, base, m) for m in ("f1", "auc", "accuracy")},
        # 搜尋 val 減去封存 test：這就是「照著 test 挑超參」會灌的水。
        "selection_optimism_f1": round(
            row["val_f1_mean"] - summarise(tuned)["mean"], 4),
        "sealed_opens": opens_a + opens_b,
        "runs": {"tuned": tuned, "baseline": base},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    c = out["comparison"]["f1"]
    print(f"\n差距 {c['gap']:+.4f}（p={c['p_value']}）→ {c['verdict']}")
    print(f"選擇性樂觀（val − test）：{out['selection_optimism_f1']:+.4f}")


if __name__ == "__main__":
    main()
