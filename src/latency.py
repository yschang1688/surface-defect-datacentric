"""延遲量測：先問「量到的是什麼」，再問「數字多少」。

產線上「這個模型夠不夠快」的答案，取決於量的是哪一件事。三個常見的假數字
在這裡全部量出來並排：

1. **批次攤提冒充單筆延遲**。`forward(batch_64)` 計時 ÷64 量到的是吞吐量的
   倒數，不是單筆延遲——它假設 64 張同時到齊。產線是一張一張來的。
2. **非同步裝置上忘記同步**。MPS／CUDA 的 kernel 是非同步派送的，
   `t0; forward(x); t1` 量到的是「把工作丟出去要多久」，不是「算完要多久」。
   不呼叫 `synchronize()` 的數字會漂亮得離譜，而且完全可重現。
3. **只算 forward、不算前處理**。真實路徑是「解碼 → 縮放 → 正規化 → 推論」，
   前處理在小模型上往往比推論還貴。

所以輸出一律標明 mode，且 `batch_amortized` 這個名字不允許被寫成 per-image
latency——`assert_claimable()` 會擋下來（有測試釘住）。

延遲數字必附量測條件：裝置、批次大小、warmup 次數、樣本數、百分位。
沒有條件的「< 50 ms」不可宣稱。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from experiment import ARCHS, MEAN, STD, build_model
from tiles import list_images

RESULTS = Path(__file__).resolve().parent.parent / "results"
WARMUP = 20
ITERS = 200

MODES = {
    "single_true": "單筆真實路徑（解碼→縮放→正規化→推論，batch=1，含同步）",
    "single_forward_only": "只計 forward 的單筆（不含前處理）",
    "single_nosync_WRONG": "單筆但未呼叫 synchronize——刻意保留的錯誤量法",
    "batch_amortized": "批次計時 ÷ 批次大小＝吞吐量的倒數，不是單筆延遲",
}
# 這些 mode 不得被寫成「單筆延遲」：它們量的是別的東西。
NOT_PER_IMAGE = {"batch_amortized"}


class ClaimError(ValueError):
    pass


def assert_claimable(mode: str, claim: str) -> None:
    """擋下「拿批次攤提當單筆延遲」這一類的宣稱。"""
    if mode not in MODES:
        raise ClaimError(f"未知量測模式：{mode}")
    if mode in NOT_PER_IMAGE and claim == "per_image_latency":
        raise ClaimError(
            f"{mode} 量的是吞吐量的倒數（假設整批同時到齊），不是單筆延遲。"
            "要報單筆請改量 single_true。")
    if mode == "single_nosync_WRONG" and claim != "demonstration":
        raise ClaimError("未同步的數字只能當反例展示，不得引用。")


def make_sync(device) -> callable:
    """非同步裝置要等它真的算完。CPU 上是 no-op，但仍走同一條路徑。"""
    if device.type == "mps":
        return torch.mps.synchronize
    if device.type == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def preprocess(path, img_size: int) -> torch.Tensor:
    """真實線上路徑：從檔案到模型輸入。這段在小模型上往往比推論還貴。"""
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)
    x = x.float() / 255.0
    return v2.Normalize(mean=MEAN, std=STD)(x).unsqueeze(0)


def percentiles(xs: list[float]) -> dict:
    a = np.array(xs) * 1000.0     # → 毫秒
    return {"p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3),
            "mean": round(float(a.mean()), 3), "n": len(xs)}


def measure(fn, sync, iters: int = ITERS, warmup: int = WARMUP) -> dict:
    """通用計時：warmup 之後逐次計時，每次都等裝置算完。

    `sync` 是參數而不是內部細節，因為「有沒有同步」正是這支程式要示範的
    差別——測試也靠它驗證同步真的被呼叫了。
    """
    for i in range(warmup):
        fn(i)
    sync()
    xs = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn(i)
        sync()
        xs.append(time.perf_counter() - t0)
    return percentiles(xs)


def profile_arch(arch: str, paths: list, device, img_size: int = 224,
                 batch: int = 64, iters: int = ITERS) -> dict:
    model = build_model(arch, 2, "none").to(device).eval()
    sync = make_sync(device)
    n_params = sum(p.numel() for p in model.parameters())
    cached = [preprocess(paths[i % len(paths)], img_size) for i in range(min(64, len(paths)))]
    batch_x = torch.cat(cached[:batch] if len(cached) >= batch
                        else cached * (batch // len(cached) + 1))[:batch].to(device)

    with torch.no_grad():
        out = {
            "arch": arch,
            "params_m": round(n_params / 1e6, 2),
            "weights_mb": round(n_params * 4 / 1024 ** 2, 1),
            "single_true": measure(
                lambda i: model(preprocess(paths[i % len(paths)], img_size).to(device)),
                sync, iters),
            "single_forward_only": measure(
                lambda i: model(cached[i % len(cached)].to(device)), sync, iters),
            # 反例：同一段程式碼，只是不等裝置算完。
            "single_nosync_WRONG": measure(
                lambda i: model(cached[i % len(cached)].to(device)),
                lambda: None, iters),
        }
        b = measure(lambda i: model(batch_x), sync, max(20, iters // 10))
        out["batch_amortized"] = {k: (round(v / batch, 4) if k != "n" else v)
                                  for k, v in b.items()} | {"batch": batch}
        out["batch_total"] = b | {"batch": batch}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out", default="latency.json")
    a = ap.parse_args()

    paths = [it["path"] for it in list_images()][:64]
    rows = []
    for dev_name in a.devices:
        if dev_name == "mps" and not torch.backends.mps.is_available():
            continue
        device = torch.device(dev_name)
        for arch in a.archs:
            r = profile_arch(arch, paths, device, iters=a.iters) | {"device": dev_name}
            rows.append(r)
            print(f"{dev_name:4s} {arch:20s} 參數 {r['params_m']:>5.2f}M｜"
                  f"單筆真實 p50 {r['single_true']['p50']:>7.2f}ms "
                  f"p99 {r['single_true']['p99']:>7.2f}｜"
                  f"只算 forward p50 {r['single_forward_only']['p50']:>6.2f}｜"
                  f"批次攤提 {r['batch_amortized']['p50']:>6.2f}｜"
                  f"未同步（錯）{r['single_nosync_WRONG']['p50']:>6.2f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / a.out).write_text(json.dumps(
        {"warmup": WARMUP, "iters": a.iters, "img_size": 224, "modes": MODES,
         "not_per_image": sorted(NOT_PER_IMAGE), "rows": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
