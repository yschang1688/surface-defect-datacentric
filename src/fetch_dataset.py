"""取得磁磚表面瑕疵資料集（Magnetic Tile Defect, Huang et al.）。

走 GitHub 免認證下載——Kaggle 一律需要帳號，而需要登入的取得路徑會讓
「完整重現」這句話對別人不成立。

資料集：6 類（Blowhole 氣孔／Break 破損／Crack 裂紋／Fray 磨損／Uneven 不均／Free 無瑕疵），
共 1,344 張灰階影像，每張附人工標註遮罩（本專案只用分類標籤，遮罩留給後續分割任務）。
"""
from __future__ import annotations

import io
import shutil
import sys
import zipfile
from pathlib import Path

import requests

URL = "https://codeload.github.com/abin24/Magnetic-tile-defect-datasets./zip/refs/heads/master"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Uneven", "Free"]


def main() -> int:
    if RAW.exists() and any(RAW.glob("MT_*/Imgs/*.jpg")):
        n = len(list(RAW.glob("MT_*/Imgs/*.jpg")))
        print(f"已存在 {n} 張影像於 {RAW}，跳過下載")
        return 0

    print("下載中（約 51 MB）…")
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
    r.raise_for_status()

    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        root = "Magnetic-tile-defect-datasets.-master/"
        for m in z.namelist():
            if not m.startswith(root) or m.endswith("/"):
                continue
            rel = Path(m[len(root):])
            if not rel.parts or not rel.parts[0].startswith("MT_"):
                continue
            dest = RAW / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    imgs = list(RAW.glob("MT_*/Imgs/*.jpg"))
    if len(imgs) < 1000:
        sys.exit(f"只解出 {len(imgs)} 張影像，來源結構可能已變")
    print(f"✓ {len(imgs)} 張影像 → {RAW}")
    for c in CLASSES:
        print(f"  MT_{c}: {len(list((RAW / f'MT_{c}' / 'Imgs').glob('*.jpg')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
