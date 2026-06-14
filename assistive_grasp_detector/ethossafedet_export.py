"""Export helpers for EthosSafeDet-A v1."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from assistive_grasp_detector.coords import letterbox_rgb_image
from assistive_grasp_detector.ethossafedet_manifest import build_calibration_manifest
from assistive_grasp_detector.ethossafedet_model import EthosSafeDetConfig, load_checkpoint_config, load_checkpoint_state, make_ethossafedet_a
from assistive_grasp_detector.schema import ETHOSSAFEDET_NUM_CLASSES


def export_onnx_reference(
    checkpoint_path: str | Path,
    output_path: str | Path,
    input_size: int = 320,
    opset: int = 13,
) -> dict[str, Any]:
    torch = _torch()
    checkpoint_config = load_checkpoint_config(str(checkpoint_path))
    model = make_ethossafedet_a(
        EthosSafeDetConfig(input_size=input_size, num_classes=ETHOSSAFEDET_NUM_CLASSES, width=checkpoint_config.width)
    )
    model.load_state_dict(load_checkpoint_state(str(checkpoint_path)))
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input_image"],
        output_names=["cls_logits", "box_ltrb"],
        opset_version=int(opset),
        do_constant_folding=True,
        dynamo=False,
    )
    return {"onnx": str(out), "input_shape": [1, 3, input_size, input_size], "outputs": ["cls_logits", "box_ltrb"]}


def export_tflite_full_int8(
    onnx_path: str | Path,
    output_path: str | Path,
    calibration_manifest: str | Path,
    input_size: int = 320,
    target_count: int | None = None,
    work_dir: str | Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    tf = _tensorflow()
    onnx = Path(onnx_path).resolve()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir) if work_dir else out.with_suffix(".saved_model")
    work.mkdir(parents=True, exist_ok=True)

    python = python_executable or sys.executable
    cmd = [python, "-m", "onnx2tf", "-i", str(onnx), "-o", str(work.resolve()), "-n"]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            "onnx2tf conversion failed\n"
            f"command: {' '.join(cmd)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    calibration = _load_or_build_calibration(calibration_manifest, target_count)
    input_layout = _saved_model_input_layout(work)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(work))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset(calibration, input_size, input_layout)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    out.write_bytes(model_bytes)
    return {
        "tflite": str(out),
        "saved_model": str(work),
        "calibration_count": len(calibration["items"]),
        "input_layout": input_layout,
    }


def _load_or_build_calibration(path: str | Path, target_count: int | None) -> dict[str, Any]:
    calibration_path = Path(path)
    text = calibration_path.read_text(encoding="utf-8")
    if not text.lstrip().startswith("{"):
        if target_count is None:
            raise ValueError("--target-count is required when passing a detector JSONL manifest as calibration source")
        generated = calibration_path.with_suffix(".calibration.json")
        return build_calibration_manifest(calibration_path, generated, target_count=target_count)
    data = json.loads(text)
    if data.get("schema_version") == "ethossafedet_calibration_v1":
        return data
    if data.get("schema_version") == "ethossafedet_manifest_v1":
        if target_count is None:
            raise ValueError("--target-count is required when passing a detector manifest as calibration source")
        generated = calibration_path.with_suffix(".calibration.json")
        return build_calibration_manifest(calibration_path, generated, target_count=target_count)
    raise ValueError(f"unsupported calibration schema: {data.get('schema_version')!r}")


def _representative_dataset(calibration: dict[str, Any], input_size: int, input_layout: str) -> Callable[[], Any]:
    def gen():
        for item in calibration.get("items", []):
            image_path = Path(str(item["image"]))
            with Image.open(image_path) as image:
                arr = np.asarray(letterbox_rgb_image(image, input_size, input_size), dtype=np.float32) / 255.0
            if input_layout == "nchw":
                arr = np.transpose(arr, (2, 0, 1))[None, ...]
            else:
                arr = arr[None, ...]
            yield [arr.astype(np.float32)]

    return gen


def _saved_model_input_layout(saved_model_dir: Path) -> str:
    tf = _tensorflow()
    loaded = tf.saved_model.load(str(saved_model_dir))
    signature = loaded.signatures.get("serving_default")
    if signature is None:
        return "nhwc"
    _, kwargs = signature.structured_input_signature
    if not kwargs:
        return "nhwc"
    tensor_spec = next(iter(kwargs.values()))
    shape = [dim if dim is not None else -1 for dim in tensor_spec.shape.as_list()]
    if len(shape) == 4 and shape[1] == 3:
        return "nchw"
    return "nhwc"


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ONNX export requires PyTorch") from exc
    return torch


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TFLite export requires TensorFlow") from exc
    return tf
