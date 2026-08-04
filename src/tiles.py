"""還原「同一片磁磚」的分組鍵。

資料集以類別分資料夾、檔名是 `exp{1..6}_num_{id}`，**沒有給磁磚身分**。
但六個 exp 組的類別分布幾乎一模一樣，抽驗也顯示跨組存在相關係數 0.95+
的近乎重複影像、且編號相鄰——同一片磁磚被拍了多次。

這件事決定了評估能不能相信：分組鍵不還原，隨機切分就會讓同一片磁磚
同時落在訓練與測試，準確率是自己看自己。所以先把身分還原出來，
再談模型。

做法：把每張影像壓成 16×16 灰階、逐張標準化後取內積當相似度，
超過門檻的連成邊，取連通分量當作「同一片磁磚」。門檻不是憑感覺定的，
見 `analyze_threshold.py` 的相似度分布。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Uneven", "Free"]
SIG_SIZE = 16
# 門檻 0.98，依據見 analyze_threshold.py 的實測分布（results/threshold_analysis.json）。
#
# 原先假設「相似度分布有谷底、取谷底」，實測打臉：0.80→0.98 是一路遞減的連續
# 分布（磁磚影像本來就長得像），**唯一的結構是 0.98–1.00 的回升峰**
# （0.96–0.98 有 1,167 對、0.98–1.00 反而有 1,963 對）——那才是近乎重複的那一群。
#
# 交叉驗證用「組內檔名編號跨距」：同一片磁磚連拍，編號應相近。
#   門檻 0.94 → 最大組 196 張（連鎖效應把不同磁磚串成一串，已經不可信）
#   門檻 0.98 → 最大組  15 張、編號跨距中位數 94（合理：一片磁磚數個 exp 各拍幾張）
#   門檻 0.99+ → 組數暴增、多張組反而變少，代表真正的重複被拆開，洩漏會殘留
# 取 0.98：寧可誤併（評估更嚴），不可漏併（漏併＝洩漏殘留＝成績灌水）。
SIM_THRESHOLD = 0.98
NAME = re.compile(r"(exp\d)_num_(\d+)\.jpg$")


def list_images() -> list[dict]:
    out = []
    for c in CLASSES:
        for p in sorted((RAW / f"MT_{c}" / "Imgs").glob("*.jpg")):
            m = NAME.search(p.name)
            if not m:
                continue
            out.append({"path": p, "label": c, "exp": m.group(1),
                        "num": int(m.group(2)), "is_defect": int(c != "Free")})
    return out


def signature(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((SIG_SIZE, SIG_SIZE))
    a = np.asarray(im, dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a.ravel() / SIG_SIZE  # 除以邊長使內積落在 [-1, 1]


def signatures(items: list[dict]) -> np.ndarray:
    return np.stack([signature(it["path"]) for it in items])


def group_ids(items: list[dict], sims: np.ndarray | None = None,
              threshold: float = SIM_THRESHOLD) -> np.ndarray:
    """連通分量分組。回傳每張影像的 group id。

    刻意只在**同類別內**連邊：跨類別的高相似度若真的存在，那是標註問題，
    不該靠分組悄悄吸收掉（那會把資料品質問題藏起來）。
    """
    n = len(items)
    if sims is None:
        S = signatures(items)
        sims = S @ S.T
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    labels = np.array([it["label"] for it in items])
    for i in range(n):
        same = np.where((sims[i] >= threshold) & (labels == labels[i]))[0]
        for j in same:
            if j > i:
                union(i, int(j))
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid
