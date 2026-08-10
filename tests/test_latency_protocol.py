"""延遲量測的協定：量錯對象的數字全都是真的，只是量到了別的東西。

三種假數字（批次攤提冒充單筆、未同步、只算 forward）都可重現、都不會報錯，
所以擋它們的方式只能是把「這個 mode 允許宣稱什麼」寫進程式並測起來。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latency import (ClaimError, MODES, NOT_PER_IMAGE,  # noqa: E402
                     assert_claimable, make_sync, measure, percentiles)


def test_batch_amortized_cannot_be_claimed_as_per_image() -> None:
    with pytest.raises(ClaimError):
        assert_claimable("batch_amortized", "per_image_latency")
    assert_claimable("batch_amortized", "throughput")      # 這個宣稱才對得上


def test_single_true_can_be_claimed_as_per_image() -> None:
    """探針：若 assert_claimable 對所有東西都丟例外，上一條的紅是假的。"""
    assert_claimable("single_true", "per_image_latency")


def test_unsynced_numbers_are_demonstration_only() -> None:
    with pytest.raises(ClaimError):
        assert_claimable("single_nosync_WRONG", "per_image_latency")
    assert_claimable("single_nosync_WRONG", "demonstration")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ClaimError):
        assert_claimable("大概三十毫秒", "per_image_latency")


def test_every_mode_has_a_written_definition() -> None:
    """沒有定義的 mode 等於沒有量測條件，數字不可引用。"""
    assert NOT_PER_IMAGE <= set(MODES)
    for mode, desc in MODES.items():
        assert desc.strip(), f"{mode} 沒有寫明量的是什麼"


def test_measure_synchronises_once_per_iteration() -> None:
    """非同步裝置上少呼叫一次 synchronize，數字會小一個量級且完全可重現。"""
    calls = {"sync": 0, "fn": 0}

    def fn(_i):
        calls["fn"] += 1

    def sync():
        calls["sync"] += 1

    measure(fn, sync, iters=10, warmup=3)
    assert calls["fn"] == 13
    assert calls["sync"] == 11, "每次計時後都要同步，另加 warmup 後的一次"


def test_measure_reports_percentiles_not_just_mean() -> None:
    """只報平均會蓋掉尾巴；產線在意的是 p99。"""
    stats = percentiles([0.001] * 90 + [0.500] * 10)
    assert stats["p99"] > stats["p50"] * 10
    assert stats["p99"] > stats["mean"], "尾巴被平均蓋掉了"
    assert set(stats) == {"p50", "p95", "p99", "mean", "n"}


def test_cpu_sync_is_a_noop_but_still_callable() -> None:
    import torch

    make_sync(torch.device("cpu"))()      # 不該炸，也不該需要 GPU


@pytest.mark.runtime_parity
def test_recorded_latency_is_internally_consistent() -> None:
    """實跑結構檢核。

    只對 MPS 斷言「單筆慢於批次攤提」：實測 CPU 上 batch=64 反而更慢
    （resnet50 攤提 105 ms vs 單筆 25 ms），因為批次要吃滿平行度才划算。
    **批次攤提不是普遍更漂亮的數字，它只是不同的數字**——把這條寫死成
    「一定更快」，等於把一個裝置相依的現象當成定律。
    """
    import json

    p = Path(__file__).resolve().parent.parent / "results" / "latency.json"
    if not p.exists():
        pytest.skip("尚未產生 results/latency.json")
    rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
    assert {r["device"] for r in rows} >= {"mps", "cpu"}, "兩種裝置都要量，否則對照不成立"
    for r in rows:
        if r["device"] == "mps":
            assert r["single_true"]["p50"] > r["batch_amortized"]["p50"], (
                f"{r['device']}/{r['arch']}：GPU 上單筆竟然比批次攤提還快，量測有問題")
        assert r["single_true"]["p50"] > r["single_forward_only"]["p50"], (
            f"{r['device']}/{r['arch']}：含前處理竟然比只算 forward 還快")
        ns, fo = r["single_nosync_WRONG"]["p50"], r["single_forward_only"]["p50"]
        if r["device"] == "mps":
            assert ns < fo, (f"{r['device']}/{r['arch']}：未同步的數字沒有偏低，"
                             "這台裝置上這個反例不成立，README 的對照要改寫")
        else:
            # CPU 是同步執行的，忘記 synchronize 不會產生假數字——這個陷阱
            # 是裝置相依的。把它寫成普遍現象，換台機器就會對不上。
            assert abs(ns - fo) / fo < 0.05, (
                f"cpu/{r['arch']}：同步與否竟然差了 {abs(ns - fo) / fo:.0%}，"
                "CPU 上不該有這個差別，量測可能受到其他負載干擾")
