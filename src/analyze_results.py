"""合併多批訓練結果，並判定「這個差距可不可以宣稱」。

核心規則：**兩個數字的差若小於量測本身的浮動，就不可以宣稱誰比較好。**
所以這裡不只印平均，還印每個協定的種子間標準差（＝雜訊帶），
再用 Welch t 檢定（不假設等變異）給出 p 值與判定。

判定字串刻意寫死三種，避免事後看著 p 值挑形容詞：
  可宣稱（p < 0.05）／有跡象但不可宣稱（0.05 ≤ p < 0.10）／落在雜訊帶內（p ≥ 0.10）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"
METRICS = ("f1", "auc", "accuracy")


def load_runs() -> list[dict]:
    runs, seen = [], set()
    for f in sorted(RESULTS.glob("training_results*.json")):
        for r in json.loads(f.read_text(encoding="utf-8"))["runs"]:
            key = (r["protocol"], r["seed"])
            if key in seen:      # 同協定同種子重複出現時只採第一份，避免灌大 n
                continue
            seen.add(key)
            runs.append(r)
    return runs


def verdict(p: float) -> str:
    if p < 0.05:
        return "可宣稱"
    if p < 0.10:
        return "有跡象但不可宣稱"
    return "落在雜訊帶內"


def main() -> None:
    runs = load_runs()
    a = [r for r in runs if r["protocol"] == "random"]
    b = [r for r in runs if r["protocol"] == "grouped"]
    print(f"隨機切分 {len(a)} 次、分組留出 {len(b)} 次\n")

    out = {"n_random": len(a), "n_grouped": len(b), "metrics": {}}
    print(f"{'指標':10s}{'隨機切分':>18s}{'分組留出':>18s}{'差距':>9s}{'p':>8s}  判定")
    for m in METRICS:
        xa = np.array([r[m] for r in a if r[m] is not None])
        xb = np.array([r[m] for r in b if r[m] is not None])
        t, p = stats.ttest_ind(xa, xb, equal_var=False)
        gap = float(xa.mean() - xb.mean())
        out["metrics"][m] = {
            "random_mean": round(float(xa.mean()), 4),
            "random_sd": round(float(xa.std(ddof=1)), 4),
            "grouped_mean": round(float(xb.mean()), 4),
            "grouped_sd": round(float(xb.std(ddof=1)), 4),
            "gap": round(gap, 4), "welch_t": round(float(t), 3),
            "p_value": round(float(p), 4), "verdict": verdict(float(p)),
        }
        d = out["metrics"][m]
        print(f"{m:10s}{d['random_mean']:>10.4f}±{d['random_sd']:.4f}"
              f"{d['grouped_mean']:>10.4f}±{d['grouped_sd']:.4f}"
              f"{gap:>9.4f}{p:>8.4f}  {d['verdict']}")

    (RESULTS / "analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {RESULTS / 'analysis.json'}")


if __name__ == "__main__":
    main()
