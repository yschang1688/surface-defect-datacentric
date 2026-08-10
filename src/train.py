"""瑕疵分類（二分類／六類擇一）：兩種切分協定 × 多顆種子。

本檔要回答的不是「準確率多少」，而是**這個準確率能不能相信**：

1. **隨機切分 vs 分組留出**。同一片磁磚的多張影像若散落在訓練與測試兩側，
   模型只要記住這片磁磚長什麼樣就能答對——量到的是記憶力不是泛化力。
   分組留出把同一片磁磚整組關進同一側，量到的才是「沒看過的磁磚」。
2. **雜訊帶**。同一協定換種子重跑多次，指標本身就會浮動。兩個數字的差
   若小於這條浮動帶，就不可以宣稱誰比較好——這是量測不確定度的常識，
   在模型評估裡卻常被忽略。

模型不是重點：ResNet-18 ImageNet 預訓練 + 線性頭微調，兩種協定完全同一套
超參數與訓練輪數，唯一的差別是資料怎麼切。

任務有兩種（`--task`），**協定的結論不因任務而改變**，切分才是本專案的主題：

- `binary`（預設）：有瑕疵／良品。README 與 results/ 既有數字全部產於此設定，
  改動預設值會讓那些數字對不上，所以預設不動。
- `multiclass`：資料集原本的六類（Blowhole／Break／Crack／Fray／Uneven／Free）。
  **這個設定的類別級結論本來就薄**——Fray 僅 32 張，切四分之一去測試只剩個位數，
  單顆種子的類別級 F1 幾乎全是抽樣雜訊。所以輸出一律附 `per_class` 的 support，
  讓「這格數字有幾張撐著」和數字本身一起被看到；support 個位數的類別不得拿來宣稱。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import BASELINE, build_transforms, fit_eval, target_of  # noqa: F401
from experiment import MEAN, STD, preload as _preload                   # noqa: F401
from tiles import CLASSES, group_ids, list_images

RESULTS = Path(__file__).resolve().parent.parent / "results"
IMG_SIZE = BASELINE.img_size
EPOCHS = BASELINE.epochs
BATCH = BASELINE.batch
LR = BASELINE.lr
TEST_FRAC = 0.25

# 訓練迴圈本身搬到 experiment.py，兩邊共用（超參搜尋若用另一套實作，
# 搜出來的贏家可能贏在實作差異上）。搬家不該換結果：
# tests/test_experiment_parity.py 釘住 BASELINE 與 results/ 既有數字相同。
TRAIN_TF, EVAL_TF, _ = build_transforms(BASELINE)


def preload(items: list[dict]) -> None:
    _preload(items, IMG_SIZE)


def split_random(items, gid, rng) -> tuple[list[int], list[int]]:
    idx = rng.permutation(len(items))
    cut = int(len(items) * TEST_FRAC)
    return idx[cut:].tolist(), idx[:cut].tolist()


def split_grouped(items, gid, rng) -> tuple[list[int], list[int]]:
    """整組進同一側；依組別抽樣直到測試集達到目標比例。"""
    groups = rng.permutation(gid.max() + 1)
    target, test_groups, n = int(len(items) * TEST_FRAC), set(), 0
    for g in groups:
        if n >= target:
            break
        test_groups.add(int(g))
        n += int((gid == g).sum())
    test = [i for i in range(len(items)) if gid[i] in test_groups]
    train = [i for i in range(len(items)) if gid[i] not in test_groups]
    return train, test


SPLITS = {"random": split_random, "grouped": split_grouped}


def run_once(items, gid, protocol: str, seed: int, device, task: str = "binary") -> dict:
    """切分照協定，訓練與評估交給 experiment.fit_eval（BASELINE 設定）。

    RNG 順序與搬家前一致：manual_seed 在 fit_eval 裡先做，而切分只吃
    numpy 的獨立亂數流，兩者互不干擾——所以同種子仍給出同一組數字。
    """
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = SPLITS[protocol](items, gid, rng)
    return {"protocol": protocol} | fit_eval(items, tr_idx, te_idx, BASELINE,
                                             seed, device, task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out", default="training_results.json")
    ap.add_argument("--protocols", nargs="+", default=["random", "grouped"])
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary",
                    help="binary＝有瑕疵／良品（預設，README 數字產於此）；"
                         "multiclass＝資料集原本的六類（Fray 僅 32 張，類別級結論薄）")
    a = ap.parse_args()
    if a.task == "multiclass" and a.out == "training_results.json":
        # 預設檔名會被 analyze_results.py 撿走與二分類結果混算，直接擋掉
        a.out = "training_results_multiclass.json"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    items = list_images()
    gid = group_ids(items)
    preload(items)
    print(f"{len(items)} 張影像 → {gid.max() + 1} 個磁磚分組｜裝置 {device}（影像已快取）")

    metrics = ("f1", "auc", "accuracy") if a.task == "binary" else ("macro_f1", "accuracy")
    rows, t0 = [], time.time()
    for protocol in a.protocols:
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            r = run_once(items, gid, protocol, seed, device, a.task)
            rows.append(r)
            if a.task == "binary":
                print(f"  {protocol:8s} seed={seed} F1={r['f1']:.4f} AUC={r['auc']} "
                      f"acc={r['accuracy']:.4f}（測試 {r['n_test']} 張，"
                      f"瑕疵率 {r['test_defect_rate']:.2f}）")
            else:
                thin = [c for c, v in r["per_class"].items() if v["support"] < 10]
                print(f"  {protocol:8s} seed={seed} macroF1={r['macro_f1']:.4f} "
                      f"acc={r['accuracy']:.4f}（測試 {r['n_test']} 張）"
                      + (f"｜support <10 的類別：{'、'.join(thin)}" if thin else ""))

    summary = {}
    for protocol in a.protocols:
        v = [r for r in rows if r["protocol"] == protocol]
        for metric in metrics:
            xs = [r[metric] for r in v if r.get(metric) is not None]
            summary.setdefault(protocol, {})[metric] = {
                "mean": round(float(np.mean(xs)), 4),
                "sd": round(float(np.std(xs, ddof=1)), 4),
                "min": round(float(np.min(xs)), 4),
                "max": round(float(np.max(xs)), 4),
            }
    if a.task == "multiclass":
        # 各類別的 support 跨種子加總：讓「這格 F1 有幾張撐著」和 F1 同時落檔，
        # 否則一個 0.00 的類別級 F1 看起來像模型爛，實際上是測試集只有 6 張。
        summary["per_class_support_total"] = {
            c: sum(r["per_class"][c]["support"] for r in rows) for c in CLASSES}
    out = {"seeds": a.seeds, "epochs": EPOCHS, "task": a.task,
           "groups": int(gid.max() + 1),
           "elapsed_sec": round(time.time() - t0, 1), "runs": rows, "summary": summary}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / a.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
