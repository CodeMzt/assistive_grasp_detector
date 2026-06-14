"""Validation gates for EthosSafeDet-A exports and runtime logs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from assistive_grasp_detector.ethossafedet_manifest import load_manifest_records, resolve_record_image
from assistive_grasp_detector.ethossafedet_postprocess import (
    candidates_to_json,
    compare_main_detection,
    decode_ltrb_outputs,
)
from assistive_grasp_detector.schema import (
    ETHOSSAFEDET_ARENA_LIMIT_BYTES,
    ETHOSSAFEDET_WEIGHTS_LIMIT_BYTES,
)


ALLOWED_ONNX_OPS = {
    "Conv",
    "Add",
    "Relu",
    "Reshape",
    "Pad",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
}
FORBIDDEN_ONNX_OPS = {
    "Sigmoid",
    "Exp",
    "Softmax",
    "NonMaxSuppression",
    "TopK",
    "ArgMax",
    "Gather",
    "Shape",
    "Range",
    "Slice",
    "Concat",
}
ALLOWED_TFLITE_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "ADD",
    "RELU",
    "RELU6",
    "RESHAPE",
    "PAD",
    "PADV2",
    "MAX_POOL_2D",
    "AVERAGE_POOL_2D",
    "QUANTIZE",
    "DEQUANTIZE",
}
ALLOWED_RUHMI_CPU_BRIDGE_OPS = {"QUANTIZE", "DEQUANTIZE", "RESHAPE", "PAD"}


def check_onnx_ops(path: str | Path, allowed_ops: set[str] | None = None) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    allowed = allowed_ops or ALLOWED_ONNX_OPS
    ops = [node.op_type for node in model.graph.node]
    unknown = sorted({op for op in ops if op not in allowed})
    forbidden = sorted({op for op in ops if op in FORBIDDEN_ONNX_OPS})
    inputs_static = [_value_info_static(value) for value in model.graph.input]
    outputs_static = [_value_info_static(value) for value in model.graph.output]
    output_names = [output.name for output in model.graph.output]
    output_contract_ok = output_names == ["cls_logits", "box_ltrb"]
    result = {
        "ok": not unknown and not forbidden and all(inputs_static) and all(outputs_static) and output_contract_ok,
        "ops": sorted(set(ops)),
        "unknown_ops": unknown,
        "forbidden_ops": forbidden,
        "inputs_static": all(inputs_static),
        "outputs_static": all(outputs_static),
        "output_names": output_names,
        "output_contract_ok": output_contract_ok,
    }
    return result


def check_tflite_ops(path: str | Path, allowed_ops: set[str] | None = None) -> dict[str, Any]:
    tf = _tensorflow()
    interpreter = _make_tflite_interpreter(tf, path)
    interpreter.allocate_tensors()
    details = interpreter._get_ops_details()  # noqa: SLF001 - no public equivalent exposes op names
    ops = [str(item.get("op_name", "")) for item in details if str(item.get("op_name", "")) != "DELEGATE"]
    allowed = allowed_ops or ALLOWED_TFLITE_OPS
    unknown = sorted({op for op in ops if op not in allowed})
    outputs = interpreter.get_output_details()
    output_count_ok = len(outputs) == 2
    result = {
        "ok": not unknown and output_count_ok,
        "ops": sorted(set(ops)),
        "unknown_ops": unknown,
        "output_count": len(outputs),
        "output_count_ok": output_count_ok,
        "outputs": [{"name": item.get("name", ""), "shape": [int(v) for v in item.get("shape", [])]} for item in outputs],
    }
    return result


def compare_onnx_tflite_reference(
    onnx_path: str | Path,
    tflite_path: str | Path,
    image_path: str | Path,
    input_size: int = 320,
    min_iou: float = 0.85,
) -> dict[str, Any]:
    onnx_outputs = run_onnx_raw(onnx_path, image_path, input_size=input_size)
    tflite_outputs = run_tflite_raw(tflite_path, image_path, input_size=input_size)
    onnx_det = decode_ltrb_outputs(onnx_outputs["cls_logits"], onnx_outputs["box_ltrb"], input_size=input_size)
    tflite_det = decode_ltrb_outputs(tflite_outputs["cls_logits"], tflite_outputs["box_ltrb"], input_size=input_size)
    comparison = compare_main_detection(candidates_to_json(onnx_det), candidates_to_json(tflite_det), min_iou=min_iou)
    return {
        "ok": comparison["ok"],
        "comparison": comparison,
        "onnx_top": candidates_to_json(onnx_det, limit=5),
        "tflite_top": candidates_to_json(tflite_det, limit=5),
    }


def run_host_mera_gate(
    reference_json: str | Path,
    host_json: str | Path | None = None,
    command: list[str] | None = None,
    min_iou: float = 0.85,
) -> dict[str, Any]:
    if host_json is None and command is None:
        raise ValueError("host_json or command is required")
    if command is not None:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            return {"ok": False, "reason": "command_failed", "returncode": completed.returncode, "stderr": completed.stderr}
        host_data = json.loads(completed.stdout)
    else:
        host_data = json.loads(Path(host_json).read_text(encoding="utf-8"))
    reference_data = json.loads(Path(reference_json).read_text(encoding="utf-8"))
    reference = _extract_detections(reference_data)
    host = _extract_detections(host_data)
    comparison = compare_main_detection(reference, host, min_iou=min_iou)
    return {"ok": comparison["ok"], "comparison": comparison}


def inspect_ruhmi_dispatch(log_path: str | Path, max_base_addr: int = 8) -> dict[str, Any]:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    base_addr = _parse_num_base_addr(text)
    cpu_violations: list[str] = []
    conv_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        upper = line.upper()
        if "CONV" in upper or "DEPTHWISE" in upper:
            conv_lines.append(line)
            if "ETHOS" not in upper and "NPU" not in upper:
                cpu_violations.append(line)
        if "CPU" in upper and not any(op in upper for op in ALLOWED_RUHMI_CPU_BRIDGE_OPS):
            if any(token in upper for token in ("CONV", "POOL", "ADD", "RELU", "SOFTMAX", "NMS", "TOPK")):
                cpu_violations.append(line)
    ok = base_addr is not None and base_addr <= max_base_addr and not cpu_violations and bool(conv_lines)
    return {
        "ok": ok,
        "num_base_addr": base_addr,
        "max_base_addr": max_base_addr,
        "conv_line_count": len(conv_lines),
        "cpu_violations": cpu_violations,
    }


def check_memory_budget(
    arena_bytes: int | None = None,
    weights_bytes: int | None = None,
    log_path: str | Path | None = None,
    arena_limit: int = ETHOSSAFEDET_ARENA_LIMIT_BYTES,
    weights_limit: int = ETHOSSAFEDET_WEIGHTS_LIMIT_BYTES,
) -> dict[str, Any]:
    if log_path is not None:
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
        arena_bytes = arena_bytes if arena_bytes is not None else _parse_bytes_metric(text, ("arena", "tensor arena"))
        weights_bytes = weights_bytes if weights_bytes is not None else _parse_bytes_metric(text, ("weights", "model weights"))
    ok = (
        arena_bytes is not None
        and weights_bytes is not None
        and int(arena_bytes) <= int(arena_limit)
        and int(weights_bytes) <= int(weights_limit)
    )
    return {
        "ok": ok,
        "arena_bytes": arena_bytes,
        "arena_limit": arena_limit,
        "weights_bytes": weights_bytes,
        "weights_limit": weights_limit,
    }


def make_static_golden(
    onnx_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    input_size: int = 320,
    limit: int = 20,
) -> dict[str, Any]:
    records = load_manifest_records(manifest_path)[:limit]
    items: list[dict[str, Any]] = []
    for record in records:
        image_path = resolve_record_image(record)
        raw = run_onnx_raw(onnx_path, image_path, input_size=input_size)
        detections = decode_ltrb_outputs(raw["cls_logits"], raw["box_ltrb"], input_size=input_size)
        items.append(
            {
                "image": image_path.as_posix(),
                "split": record.get("split", ""),
                "detections": candidates_to_json(detections),
            }
        )
    golden = {
        "schema_version": "ethossafedet_static_golden_v1",
        "onnx": Path(onnx_path).resolve().as_posix(),
        "input_size": input_size,
        "items": items,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return golden


def run_onnx_raw(onnx_path: str | Path, image_path: str | Path, input_size: int = 320) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    arr = _load_nchw_image(image_path, input_size)
    outputs = session.run(["cls_logits", "box_ltrb"], {input_name: arr})
    return {"cls_logits": outputs[0], "box_ltrb": outputs[1]}


def run_tflite_raw(tflite_path: str | Path, image_path: str | Path, input_size: int = 320) -> dict[str, np.ndarray]:
    tf = _tensorflow()
    interpreter = _make_tflite_interpreter(tf, tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    arr = _load_image_for_tflite(image_path, input_size, inp)
    interpreter.set_tensor(inp["index"], arr)
    interpreter.invoke()
    outputs = []
    for out in interpreter.get_output_details():
        value = interpreter.get_tensor(out["index"])
        quant = out.get("quantization", (0.0, 0))
        scale, zero_point = float(quant[0]), int(quant[1])
        if scale > 0:
            value = (value.astype(np.float32) - float(zero_point)) * scale
        outputs.append(value)
    return _split_cls_box_outputs(outputs)


def write_json_result(result: dict[str, Any], output_path: str | Path | None = None) -> None:
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _value_info_static(value_info: Any) -> bool:
    shape = value_info.type.tensor_type.shape
    for dim in shape.dim:
        if not dim.HasField("dim_value"):
            return False
    return True


def _load_nchw_image(image_path: str | Path, input_size: int) -> np.ndarray:
    from PIL import Image

    from assistive_grasp_detector.coords import letterbox_rgb_image

    with Image.open(image_path) as image:
        arr = np.asarray(letterbox_rgb_image(image, input_size, input_size), dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def _load_image_for_tflite(image_path: str | Path, input_size: int, input_detail: dict[str, Any]) -> np.ndarray:
    arr_nchw = _load_nchw_image(image_path, input_size)
    shape = [int(v) for v in input_detail["shape"]]
    if len(shape) == 4 and shape[1] == 3:
        arr = arr_nchw
    else:
        arr = np.transpose(arr_nchw, (0, 2, 3, 1))
    dtype = input_detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if float(scale) <= 0:
            raise ValueError("quantized TFLite input has invalid scale")
        arr = np.round(arr / float(scale) + int(zero_point)).astype(dtype)
    else:
        arr = arr.astype(dtype)
    return arr


def _extract_detections(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("detections"), list):
        return data["detections"]
    items = data.get("items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0].get("detections", [])
    raise ValueError("cannot find detections in JSON")


def _split_cls_box_outputs(outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    if len(outputs) != 2:
        raise ValueError(f"expected 2 TFLite outputs, got {len(outputs)}")
    cls = None
    box = None
    for value in outputs:
        shape = list(value.shape)
        channels = None
        if len(shape) == 4:
            if shape[1] in (4, 6):
                channels = shape[1]
            elif shape[-1] in (4, 6):
                channels = shape[-1]
        if channels == 6:
            cls = value
        elif channels == 4:
            box = value
    if cls is None or box is None:
        raise ValueError(f"cannot infer cls/box outputs from shapes {[list(v.shape) for v in outputs]}")
    return {"cls_logits": cls, "box_ltrb": box}


def _parse_num_base_addr(text: str) -> int | None:
    match = re.search(r"num_base_addr\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_bytes_metric(text: str, names: tuple[str, ...]) -> int | None:
    for name in names:
        pattern = rf"{re.escape(name)}[^\d]*(\d+(?:\.\d+)?)\s*(mib|mb|kib|kb|bytes|b)?"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = (match.group(2) or "bytes").lower()
            if unit == "mib":
                value *= 1024 * 1024
            elif unit == "mb":
                value *= 1000 * 1000
            elif unit == "kib":
                value *= 1024
            elif unit == "kb":
                value *= 1000
            return int(value)
    return None


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TFLite gates require TensorFlow") from exc
    return tf


def _make_tflite_interpreter(tf, path: str | Path):  # type: ignore[no-untyped-def]
    kwargs: dict[str, Any] = {"model_path": str(path), "experimental_delegates": []}
    experimental = getattr(tf.lite, "experimental", None)
    resolver_type = getattr(experimental, "OpResolverType", None) if experimental is not None else None
    if resolver_type is not None and hasattr(resolver_type, "BUILTIN_REF"):
        kwargs["experimental_op_resolver_type"] = resolver_type.BUILTIN_REF
    try:
        return tf.lite.Interpreter(**kwargs, experimental_preserve_all_tensors=True)
    except TypeError:
        return tf.lite.Interpreter(**kwargs)
