"""四個架構的接線：分類頭掛的位置與凍結的範圍，錯了不會報錯。

`model.fc` 對 ResNet 成立、對 MobileNet 不成立；`freeze="backbone"` 在
ResNet 上是 512→2 的線性探針（0.001M 參數），但 MobileNet 的 `classifier`
裡還藏了一層 Linear(576→1024)——整段解凍就有 0.59M 參數在訓練，
那已經不是線性探針，而是一個五百倍大的模型在跟別人比。

架構對照的前提是「除了 backbone 以外都一樣」，所以這裡把
「一樣」這件事測出來，而不是相信它。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import ARCHS, build_model  # noqa: E402


def trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_head_outputs_requested_classes(arch: str) -> None:
    for n_out in (2, 6):
        model = build_model(arch, n_out, "none").eval()
        with torch.no_grad():
            assert model(torch.randn(2, 3, 224, 224)).shape == (2, n_out)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_linear_probe_trains_only_the_final_layer(arch: str) -> None:
    """freeze=backbone 在每個架構上都必須是同一件事：只訓練最後那層。"""
    model = build_model(arch, 2, "backbone")
    n = trainable(model)
    assert n < 5000, (
        f"{arch} 的線性探針有 {n} 個可訓練參數——頭部抓錯層了，"
        "這個架構在對照裡會有主場優勢。")
    assert n > 0, f"{arch} 的線性探針什麼都沒解凍，訓練會完全無效"


@pytest.mark.parametrize("arch", list(ARCHS))
def test_freeze_levels_are_ordered(arch: str) -> None:
    """探針：三種凍結若沒真的分級，上面兩條可能同時假綠。"""
    probe = trainable(build_model(arch, 2, "backbone"))
    partial = trainable(build_model(arch, 2, "partial"))
    full = trainable(build_model(arch, 2, "none"))
    assert probe < partial < full, f"{arch}：backbone < partial < none 的順序不成立"


@pytest.mark.parametrize("arch", list(ARCHS))
def test_frozen_params_really_have_no_grad(arch: str) -> None:
    """requires_grad=False 若只設在 optimizer 那側，權重照樣會被更新。"""
    model = build_model(arch, 2, "backbone")
    model(torch.randn(2, 3, 224, 224)).sum().backward()
    frozen_with_grad = [n for n, p in model.named_parameters()
                        if not p.requires_grad and p.grad is not None]
    assert not frozen_with_grad, f"{arch}：凍結參數仍拿到梯度 {frozen_with_grad[:3]}"
