"""EthosSafeDetV2 model definition for Model A formal training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from assistive_grasp_detector.schema import ETHOSSAFEDET_NUM_CLASSES


@dataclass(frozen=True)
class EthosSafeDetV2Config:
    input_size: int = 320
    num_classes: int = ETHOSSAFEDET_NUM_CLASSES
    width: int = 40


V2_OUTPUT_NAMES = [
    "s8_cls_logits",
    "s8_box_ltrb",
    "s8_orientation",
    "s16_cls_logits",
    "s16_box_ltrb",
    "s16_orientation",
]


def make_ethossafedet_v2(config: EthosSafeDetV2Config | None = None):
    torch, nn = _torch_modules()
    cfg = config or EthosSafeDetV2Config()
    return EthosSafeDetV2(nn=nn, config=cfg)


class EthosSafeDetV2:
    """Factory wrapper so importing this module does not require torch."""

    def __new__(cls, nn, config: EthosSafeDetV2Config):  # type: ignore[no-untyped-def]
        class ConvRelu(nn.Module):  # type: ignore[misc]
            def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> None:
                super().__init__()
                self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, groups=groups, bias=True)
                self.relu = nn.ReLU(inplace=False)

            def forward(self, x):  # type: ignore[no-untyped-def]
                return self.relu(self.conv(x))

        class DepthwiseBlock(nn.Module):  # type: ignore[misc]
            def __init__(self, in_ch: int, out_ch: int, stride: int = 1, residual: bool = False) -> None:
                super().__init__()
                self.residual = bool(residual and stride == 1 and in_ch == out_ch)
                self.dw = ConvRelu(in_ch, in_ch, kernel=3, stride=stride, groups=in_ch)
                self.pw = ConvRelu(in_ch, out_ch, kernel=1, stride=1, groups=1)

            def forward(self, x):  # type: ignore[no-untyped-def]
                y = self.pw(self.dw(x))
                if self.residual:
                    y = y + x
                return y

        class Head(nn.Module):  # type: ignore[misc]
            def __init__(self, in_ch: int, hidden_ch: int, num_classes: int) -> None:
                super().__init__()
                self.cls_tower = DepthwiseBlock(in_ch, hidden_ch, stride=1, residual=False)
                self.box_tower = DepthwiseBlock(in_ch, hidden_ch, stride=1, residual=False)
                self.ori_tower = DepthwiseBlock(in_ch, hidden_ch, stride=1, residual=False)
                self.cls_head = nn.Conv2d(hidden_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                self.box_head = nn.Conv2d(hidden_ch, 4, kernel_size=1, stride=1, padding=0, bias=True)
                self.ori_head = nn.Conv2d(hidden_ch, 2, kernel_size=1, stride=1, padding=0, bias=True)
                self.box_relu = nn.ReLU(inplace=False)
                nn.init.constant_(self.box_head.bias, 16.0)

            def forward(self, x):  # type: ignore[no-untyped-def]
                return (
                    self.cls_head(self.cls_tower(x)),
                    self.box_relu(self.box_head(self.box_tower(x))),
                    self.ori_head(self.ori_tower(x)),
                )

        class Model(nn.Module):  # type: ignore[misc]
            def __init__(self, cfg: EthosSafeDetV2Config) -> None:
                super().__init__()
                w = int(cfg.width)
                self.config = cfg
                self.stem = ConvRelu(3, w, stride=2)
                self.block1 = DepthwiseBlock(w, w * 2, stride=2)
                self.block2 = DepthwiseBlock(w * 2, w * 3, stride=2)
                self.block3 = DepthwiseBlock(w * 3, w * 3, stride=1, residual=True)
                self.block4 = DepthwiseBlock(w * 3, w * 4, stride=2)
                self.block5 = DepthwiseBlock(w * 4, w * 4, stride=1, residual=True)
                self.block6 = DepthwiseBlock(w * 4, w * 4, stride=1, residual=True)
                self.s8_head = Head(w * 3, w * 3, cfg.num_classes)
                self.s16_head = Head(w * 4, w * 4, cfg.num_classes)

            def forward(self, x):  # type: ignore[no-untyped-def]
                x = self.stem(x)
                x = self.block1(x)
                s8 = self.block2(x)
                s8 = self.block3(s8)
                s16 = self.block4(s8)
                s16 = self.block5(s16)
                s16 = self.block6(s16)
                s8_cls, s8_box, s8_ori = self.s8_head(s8)
                s16_cls, s16_box, s16_ori = self.s16_head(s16)
                return s8_cls, s8_box, s8_ori, s16_cls, s16_box, s16_ori

        return Model(config)


def load_v2_checkpoint(path: str):
    torch, _ = _torch_modules()
    return torch.load(path, map_location="cpu", weights_only=False)


def load_v2_checkpoint_state(path: str) -> dict[str, Any]:
    data = load_v2_checkpoint(path)
    if isinstance(data, dict) and "model_state" in data:
        return data["model_state"]
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data


def load_v2_checkpoint_config(path: str) -> EthosSafeDetV2Config:
    data = load_v2_checkpoint(path)
    if not isinstance(data, dict):
        return EthosSafeDetV2Config()
    raw = data.get("config", data)
    return EthosSafeDetV2Config(
        input_size=int(raw.get("input_size", data.get("input_size", EthosSafeDetV2Config.input_size))),
        num_classes=int(raw.get("num_classes", data.get("num_classes", EthosSafeDetV2Config.num_classes))),
        width=int(raw.get("width", data.get("width", EthosSafeDetV2Config.width))),
    )


def parameter_count(model) -> int:  # type: ignore[no-untyped-def]
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on local ML env
        raise RuntimeError("EthosSafeDetV2 model operations require PyTorch") from exc
    return torch, nn
