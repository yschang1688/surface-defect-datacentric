"""來源端影像品質閘：把「哪些影像不該進訓練集」訂在資料進門的地方。

這套規則不是通用的影像檢查，而是**這個量測任務的方法學前提**——就像
色譜的校正曲線過期、DSC 的 Tg 沒註明升溫速率，資料進了庫就再也看不出來。

三級判定（Pass／Conditional／Reject）而非二分法：現場資料很少非黑即白，
把「可疑但可用」硬歸成任一端，不是丟掉真資料就是放進髒資料。
原始檔一律不刪不改，判定另存，隨時可回溯。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

from tiles import list_images

RESULTS = Path(__file__).resolve().parent.parent / "results"

MIN_SIDE = 64            # 短邊過小：瑕疵特徵在縮放到 224 後會被內插抹平
MAX_ASPECT = 6.0         # 極端長寬比：等比縮放後嚴重變形
DARK_MEAN, BRIGHT_MEAN = 20.0, 235.0   # 曝光失敗，細節被壓在端點
LOW_CONTRAST_SD = 8.0    # 對比過低：訊號被雜訊淹沒
BLUR_LAP_VAR = 15.0      # 拉普拉斯變異數過低＝離焦，紋理類瑕疵不可判


@dataclass
class Verdict:
    path: str
    label: str
    verdict: str          # PASS / CONDITIONAL / REJECT
    rules: list[str]


def laplacian_var(a: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = a.shape
    if h < 3 or w < 3:
        return 0.0
    win = np.lib.stride_tricks.sliding_window_view(a, (3, 3))
    return float((win * k).sum(axis=(-2, -1)).var())


def check(path: Path, label: str) -> Verdict:
    im = Image.open(path).convert("L")
    a = np.asarray(im, dtype=np.float32)
    w, h = im.size
    hard, soft = [], []

    if min(w, h) < MIN_SIDE:
        hard.append(f"SIZE_TOO_SMALL(短邊{min(w, h)}<{MIN_SIDE})")
    if max(w, h) / max(1, min(w, h)) > MAX_ASPECT:
        hard.append(f"EXTREME_ASPECT({max(w, h) / min(w, h):.1f}>{MAX_ASPECT})")
    if a.mean() < DARK_MEAN:
        hard.append(f"UNDEREXPOSED(均值{a.mean():.1f})")
    if a.mean() > BRIGHT_MEAN:
        hard.append(f"OVEREXPOSED(均值{a.mean():.1f})")

    if a.std() < LOW_CONTRAST_SD:
        soft.append(f"LOW_CONTRAST(sd{a.std():.1f})")
    lv = laplacian_var(a)
    if lv < BLUR_LAP_VAR:
        soft.append(f"POSSIBLY_BLURRED(lapvar{lv:.1f})")

    verdict = "REJECT" if hard else ("CONDITIONAL" if soft else "PASS")
    return Verdict(str(path.relative_to(path.parents[3])), label, verdict, hard + soft)


def main() -> int:
    items = list_images()
    verdicts = [check(it["path"], it["label"]) for it in items]
    counts: dict[str, int] = {}
    rules: dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
        for r in v.rules:
            rules[r.split("(")[0]] = rules.get(r.split("(")[0], 0) + 1

    print(f"{len(verdicts)} 張影像")
    for k in ("PASS", "CONDITIONAL", "REJECT"):
        print(f"  {k:12s} {counts.get(k, 0):5d}")
    print("觸發規則：" + ("、".join(f"{k}×{v}" for k, v in sorted(rules.items())) or "無"))

    # 逐類別的 CONDITIONAL 比例——若某一類特別高，那是取像條件的系統性差異，
    # 不是隨機雜訊；模型可能因此學到「這一類比較模糊」而非瑕疵本身。
    by_label: dict[str, list[int]] = {}
    for v in verdicts:
        by_label.setdefault(v.label, []).append(v.verdict != "PASS")
    print("各類非 PASS 比例：" + "、".join(
        f"{k} {np.mean(v):.0%}" for k, v in sorted(by_label.items())))

    # 一個從不作響的閘門，跟壞掉的閘門看起來一模一樣。所以把每條規則所依據的
    # 量測值分布印出來，讓「沒觸發」是可查證的結論（門檻離資料多遠），
    # 而不是一句空話。閘門本身會不會響，另由 tests/test_quality_gate.py 注入證明。
    stats = {"min_side": [], "aspect": [], "mean": [], "sd": [], "lap_var": []}
    for it in items:
        im = Image.open(it["path"]).convert("L")
        a = np.asarray(im, dtype=np.float32)
        w, h = im.size
        stats["min_side"].append(min(w, h))
        stats["aspect"].append(max(w, h) / max(1, min(w, h)))
        stats["mean"].append(a.mean())
        stats["sd"].append(a.std())
        stats["lap_var"].append(laplacian_var(a))
    dist = {k: {"p1": round(float(np.percentile(v, 1)), 2),
                "p50": round(float(np.percentile(v, 50)), 2),
                "p99": round(float(np.percentile(v, 99)), 2)}
            for k, v in stats.items()}
    print("\n量測值分布（p1／中位數／p99）與門檻距離：")
    for k, th in (("min_side", f"≥{MIN_SIDE}"), ("aspect", f"≤{MAX_ASPECT}"),
                  ("mean", f"{DARK_MEAN}–{BRIGHT_MEAN}"), ("sd", f"≥{LOW_CONTRAST_SD}"),
                  ("lap_var", f"≥{BLUR_LAP_VAR}")):
        d = dist[k]
        print(f"  {k:9s} {d['p1']:>8} / {d['p50']:>8} / {d['p99']:>8}   門檻 {th}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "quality_gate.json").write_text(
        json.dumps({"counts": counts, "rules": rules, "distribution": dist,
                    "verdicts": [asdict(v) for v in verdicts]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {RESULTS / 'quality_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
