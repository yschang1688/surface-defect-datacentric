"""相似度門檻不能憑感覺定——先看分布再定，並把依據留下來。

輸出：同類別內所有影像對的相似度直方圖，以及不同門檻下的分組結果
（組數、最大組、跨 exp 組的比例）。門檻選在分布谷底偏保守側。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tiles import group_ids, list_images, signatures

OUT = Path(__file__).resolve().parent.parent / "results" / "threshold_analysis.json"


def main() -> None:
    items = list_images()
    S = signatures(items)
    sims = S @ S.T
    labels = np.array([it["label"] for it in items])
    n = len(items)

    iu = np.triu_indices(n, k=1)
    same_label = labels[iu[0]] == labels[iu[1]]
    pair_sims = sims[iu][same_label]

    bins = np.arange(0.80, 1.001, 0.02)
    hist, _ = np.histogram(pair_sims, bins=bins)
    print("同類別影像對的相似度分布（找谷底）")
    for lo, hi, c in zip(bins[:-1], bins[1:], hist):
        print(f"  {lo:.2f}–{hi:.2f}: {c:6d} {'#' * min(60, c // 40)}")

    rows = []
    for th in (0.94, 0.96, 0.98, 0.99, 0.995):
        gid = group_ids(items, sims=sims, threshold=th)
        sizes = np.bincount(gid)
        cross = sum(
            1 for g in range(gid.max() + 1)
            if len({items[i]["exp"] for i in np.where(gid == g)[0]}) > 1)
        # 交叉驗證：同一片磁磚連拍，檔名編號應該相近。組內編號跨距的中位數
        # 若隨門檻放寬而暴衝，代表連鎖效應把不同磁磚串在一起了。
        spans = []
        for g in range(gid.max() + 1):
            idx = np.where(gid == g)[0]
            if len(idx) > 1:
                nums = sorted(items[i]["num"] for i in idx)
                spans.append(nums[-1] - nums[0])
        rows.append({"threshold": th, "groups": int(gid.max() + 1),
                     "largest_group": int(sizes.max()),
                     "multi_image_groups": int((sizes > 1).sum()),
                     "cross_exp_groups": cross,
                     "median_num_span": int(np.median(spans)) if spans else 0})
        print(f"門檻 {th}: {rows[-1]['groups']:4d} 組、最大組 {rows[-1]['largest_group']:4d} 張、"
              f"多張組 {rows[-1]['multi_image_groups']:4d}、跨 exp 組 {cross:4d}、"
              f"組內編號跨距中位數 {rows[-1]['median_num_span']:,}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"images": n, "sweep": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
