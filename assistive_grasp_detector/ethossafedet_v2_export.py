"""ONNX export for EthosSafeDetV2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from assistive_grasp_detector.ethossafedet_v2_model import (
    V2_OUTPUT_NAMES,
    load_v2_checkpoint_config,
    load_v2_checkpoint_state,
    make_ethossafedet_v2,
    parameter_count,
)


def export_v2_onnx(checkpoint: str | Path, output_path: str | Path, input_size: int | None = None, opset: int = 13) -> dict[str, Any]:
    torch, _ = _torch_modules()
    checkpoint_path = Path(checkpoint)
    cfg = load_v2_checkpoint_config(str(checkpoint_path))
    if input_size is not None:
        cfg = type(cfg)(input_size=int(input_size), num_classes=cfg.num_classes, width=cfg.width)
    model = make_ethossafedet_v2(cfg)
    model.load_state_dict(load_v2_checkpoint_state(str(checkpoint_path)))
    model.eval()
    dummy = torch.zeros((1, 3, int(cfg.input_size), int(cfg.input_size)), dtype=torch.float32)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=V2_OUTPUT_NAMES,
        opset_version=int(opset),
        do_constant_folding=True,
        dynamic_axes=None,
    )
    return {
        "onnx": out.resolve().as_posix(),
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "input_size": int(cfg.input_size),
        "width": int(cfg.width),
        "parameter_count": parameter_count(model),
        "estimated_int8_weight_bytes": parameter_count(model),
        "output_names": list(V2_OUTPUT_NAMES),
    }


def _torch_modules():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ONNX export requires PyTorch") from exc
    return torch, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export EthosSafeDetV2 ONNX reference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    try:
        result = export_v2_onnx(args.checkpoint, args.out, input_size=args.input_size, opset=args.opset)
    except Exception as exc:
        print(str(exc))
        return 1
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({"ok": True, **result}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
