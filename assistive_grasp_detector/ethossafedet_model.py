"""EthosSafeDet-A v1 PyTorch model definition.

The module intentionally uses only convolution, depthwise convolution, Add, and
ReLU-style blocks in the inference graph. Decode, sigmoid, thresholding, and
NMS are CPU-side post-processing steps.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistive_grasp_detector.schema import ETHOSSAFEDET_NUM_CLASSES


@dataclass(frozen=True)
class EthosSafeDetConfig:
    input_size: int = 320
    num_classes: int = ETHOSSAFEDET_NUM_CLASSES
    width: int = 32


def make_ethossafedet_a(config: EthosSafeDetConfig | None = None):
    torch, nn = _torch_modules()
    cfg = config or EthosSafeDetConfig()
    return EthosSafeDetA(nn=nn, config=cfg)


class EthosSafeDetA:
    """Factory wrapper so importing this module does not require torch."""

    def __new__(cls, nn, config: EthosSafeDetConfig):  # type: ignore[no-untyped-def]
        torch, _ = _torch_modules()

        class ConvRelu(nn.Module):  # type: ignore[misc]
            def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> None:
                super().__init__()
                pad = kernel // 2
                self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, groups=groups, bias=True)
                self.relu = nn.ReLU(inplace=False)

            def forward(self, x):  # type: ignore[no-untyped-def]
                return self.relu(self.conv(x))

        class DepthwiseBlock(nn.Module):  # type: ignore[misc]
            def __init__(self, in_ch: int, out_ch: int, stride: int = 1, residual: bool = False) -> None:
                super().__init__()
                self.residual = residual and stride == 1 and in_ch == out_ch
                self.dw = ConvRelu(in_ch, in_ch, kernel=3, stride=stride, groups=in_ch)
                self.pw = ConvRelu(in_ch, out_ch, kernel=1, stride=1, groups=1)

            def forward(self, x):  # type: ignore[no-untyped-def]
                y = self.pw(self.dw(x))
                if self.residual:
                    y = y + x
                return y

        class Model(nn.Module):  # type: ignore[misc]
            def __init__(self, cfg: EthosSafeDetConfig) -> None:
                super().__init__()
                w = int(cfg.width)
                self.config = cfg
                self.stem = ConvRelu(3, w, stride=2)
                self.block1 = DepthwiseBlock(w, w * 2, stride=2)
                self.block2 = DepthwiseBlock(w * 2, w * 3, stride=2)
                self.block3 = DepthwiseBlock(w * 3, w * 3, stride=1, residual=True)
                self.block4 = DepthwiseBlock(w * 3, w * 4, stride=1)
                self.block5 = DepthwiseBlock(w * 4, w * 4, stride=1, residual=True)
                self.block6 = DepthwiseBlock(w * 4, w * 4, stride=1, residual=True)
                self.block7 = DepthwiseBlock(w * 4, w * 4, stride=1, residual=True)
                self.cls_tower = DepthwiseBlock(w * 4, w * 4, stride=1, residual=True)
                self.box_tower = DepthwiseBlock(w * 4, w * 4, stride=1, residual=True)
                self.cls_head = nn.Conv2d(w * 4, cfg.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                self.box_head = nn.Conv2d(w * 4, 4, kernel_size=1, stride=1, padding=0, bias=True)
                self.box_relu = nn.ReLU(inplace=False)
                nn.init.constant_(self.box_head.bias, 16.0)

            def forward(self, x):  # type: ignore[no-untyped-def]
                x = self.stem(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.block4(x)
                x = self.block5(x)
                x = self.block6(x)
                x = self.block7(x)
                cls_logits = self.cls_head(self.cls_tower(x))
                box_ltrb = self.box_relu(self.box_head(self.box_tower(x)))
                return cls_logits, box_ltrb

        return Model(config)


def load_checkpoint_state(path: str):
    torch, _ = _torch_modules()
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "model_state" in data:
        return data["model_state"]
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data


def load_checkpoint_config(path: str) -> EthosSafeDetConfig:
    torch, _ = _torch_modules()
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        return EthosSafeDetConfig()
    return EthosSafeDetConfig(
        input_size=int(data.get("input_size", EthosSafeDetConfig.input_size)),
        num_classes=int(data.get("num_classes", EthosSafeDetConfig.num_classes)),
        width=int(data.get("width", EthosSafeDetConfig.width)),
    )


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on local ML env
        raise RuntimeError("EthosSafeDet model operations require PyTorch") from exc
    return torch, nn
