"""Static gates for EthosSafeDetV2 ONNX and artifact budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from assistive_grasp_detector.ethossafedet_v2_model import V2_OUTPUT_NAMES

ALLOWED_ONNX_OPS = {
    "Add",
    "Conv",
    "Relu",
    "Constant",
    "Identity",
}
FORBIDDEN_ONNX_OPS = {
    "Sigmoid",
    "Softmax",
    "TopK",
    "NonMaxSuppression",
    "Gather",
    "Shape",
    "Range",
    "Exp",
}


def check_v2_onnx_ops(path: str | Path, input_size: int = 320) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    ops = [node.op_type for node in model.graph.node]
    unknown = sorted({op for op in ops if op not in ALLOWED_ONNX_OPS})
    forbidden = sorted({op for op in ops if op in FORBIDDEN_ONNX_OPS})
    output_names = [output.name for output in model.graph.output]
    output_shapes = {output.name: _value_shape(output) for output in model.graph.output}
    expected_shapes = {
        "s8_cls_logits": [1, 7, input_size // 8, input_size // 8],
        "s8_box_ltrb": [1, 4, input_size // 8, input_size // 8],
        "s8_orientation": [1, 2, input_size // 8, input_size // 8],
        "s16_cls_logits": [1, 7, input_size // 16, input_size // 16],
        "s16_box_ltrb": [1, 4, input_size // 16, input_size // 16],
        "s16_orientation": [1, 2, input_size // 16, input_size // 16],
    }
    output_names_ok = output_names == V2_OUTPUT_NAMES
    output_shapes_ok = all(output_shapes.get(name) == shape for name, shape in expected_shapes.items())
    inputs_static = all(_shape_static(_value_shape(value)) for value in model.graph.input)
    outputs_static = all(_shape_static(shape) for shape in output_shapes.values())
    return {
        "ok": not unknown and not forbidden and output_names_ok and output_shapes_ok and inputs_static and outputs_static,
        "ops": sorted(set(ops)),
        "unknown_ops": unknown,
        "forbidden_ops": forbidden,
        "output_names": output_names,
        "output_names_ok": output_names_ok,
        "output_shapes": output_shapes,
        "expected_output_shapes": expected_shapes,
        "output_shapes_ok": output_shapes_ok,
        "inputs_static": inputs_static,
        "outputs_static": outputs_static,
    }


def check_v2_weight_budget(train_report: str | Path, min_int8_bytes: int = 100 * 1024, max_int8_bytes: int = 300 * 1024) -> dict[str, Any]:
    report = json.loads(Path(train_report).read_text(encoding="utf-8"))
    model = report.get("model", {})
    int8_bytes = int(model.get("estimated_int8_weight_bytes", 0))
    fp32_bytes = int(model.get("estimated_fp32_weight_bytes", 0))
    return {
        "ok": min_int8_bytes <= int8_bytes <= max_int8_bytes,
        "estimated_int8_weight_bytes": int8_bytes,
        "estimated_fp32_weight_bytes": fp32_bytes,
        "min_int8_bytes": int(min_int8_bytes),
        "max_int8_bytes": int(max_int8_bytes),
        "parameter_count": int(model.get("parameter_count", 0)),
    }


def write_gate(path: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _value_shape(value: Any) -> list[int | str]:
    dims = []
    tensor_type = value.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append("?")
    return dims


def _shape_static(shape: list[int | str]) -> bool:
    return all(isinstance(dim, int) and dim > 0 for dim in shape)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EthosSafeDetV2 static gates.")
    sub = parser.add_subparsers(dest="command", required=True)
    onnx_p = sub.add_parser("onnx")
    onnx_p.add_argument("--onnx", required=True)
    onnx_p.add_argument("--input-size", type=int, default=320)
    onnx_p.add_argument("--out-json", required=True)
    budget_p = sub.add_parser("budget")
    budget_p.add_argument("--train-report", required=True)
    budget_p.add_argument("--out-json", required=True)
    args = parser.parse_args(argv)
    if args.command == "onnx":
        result = check_v2_onnx_ops(args.onnx, input_size=args.input_size)
    else:
        result = check_v2_weight_budget(args.train_report)
    write_gate(args.out_json, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
