"""Command line entry points for EthosSafeDet-A infrastructure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from assistive_grasp_detector.annotator_dataset import validate_self_dataset
from assistive_grasp_detector.ethossafedet_export import export_onnx_reference, export_tflite_full_int8
from assistive_grasp_detector.ethossafedet_gates import (
    check_memory_budget,
    check_onnx_ops,
    check_tflite_ops,
    compare_onnx_tflite_reference,
    inspect_ruhmi_dispatch,
    make_static_golden,
    run_host_mera_gate,
    write_json_result,
)
from assistive_grasp_detector.ethossafedet_manifest import (
    build_calibration_manifest,
    prepare_ethossafedet_manifest_from_export,
    prepare_ethossafedet_manifest_from_self_dataset,
)
from assistive_grasp_detector.ethossafedet_report import make_formal_chain_report
from assistive_grasp_detector.ethossafedet_train import train_ethossafedet_a


def validate_self_dataset_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an AssistiveGraspAnnotator dataset.")
    parser.add_argument("--dataset", required=True, help="Path to the annotator dataset root.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    report = validate_self_dataset(args.dataset)
    _print_result(report.to_dict(), as_json=args.json)
    return 0 if report.ok else 1


def prepare_ethossafedet_manifest_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an EthosSafeDet-A JSONL manifest.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, help="Output ethossafedet_manifest.jsonl path.")
    parser.add_argument("--source-format", choices=["export", "self"], default="export")
    parser.add_argument("--image-subdir", default="camera_1")
    parser.add_argument("--label-subdir", default="camera_1")
    parser.add_argument("--negative-image-id", action="append", default=[])
    parser.add_argument("--allow-non-v1-classes", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    strict = not args.allow_non_v1_classes
    if args.source_format == "self":
        result = prepare_ethossafedet_manifest_from_self_dataset(args.dataset, args.out, strict_classes=strict)
    else:
        result = prepare_ethossafedet_manifest_from_export(
            args.dataset,
            args.out,
            negative_image_ids=args.negative_image_id,
            image_subdir=args.image_subdir,
            label_subdir=args.label_subdir,
            strict_classes=strict,
        )
    _print_result(result.to_dict(), as_json=args.json)
    return 0 if result.ok else 1


def build_ethossafedet_calibration_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample a 200-500 image EthosSafeDet calibration manifest.")
    parser.add_argument("--manifest", required=True, help="Source EthosSafeDet JSONL manifest.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-count", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = build_calibration_manifest(args.manifest, args.out, target_count=args.target_count, seed=args.seed)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_result({"ok": True, "output_path": args.out, "count": len(result["items"])}, as_json=True)
    return 0


def train_ethossafedet_a_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train EthosSafeDet-A v1 from a JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-cache-images", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    args = parser.parse_args(argv)
    try:
        result = train_ethossafedet_a(
            args.manifest,
            args.out,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            eval_score_threshold=args.eval_score_threshold,
            nms_iou_threshold=args.nms_iou,
            eval_every=args.eval_every,
            eval_limit=args.eval_limit,
            amp=args.amp,
            cache_images=not args.no_cache_images,
            resume_checkpoint=args.resume_checkpoint,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_result({"ok": True, **result}, as_json=True)
    return 0


def export_ethossafedet_onnx_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export EthosSafeDet-A ONNX PC reference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args(argv)
    try:
        result = export_onnx_reference(args.checkpoint, args.out, input_size=args.input_size, opset=args.opset)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_result({"ok": True, **result}, as_json=True)
    return 0


def export_ethossafedet_tflite_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export EthosSafeDet-A TFLite full-int8 from ONNX.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args(argv)
    try:
        result = export_tflite_full_int8(
            args.onnx,
            args.out,
            args.calibration,
            input_size=args.input_size,
            target_count=args.target_count,
            work_dir=args.work_dir,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_result({"ok": True, **result}, as_json=True)
    return 0


def check_onnx_ops_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)
    result = check_onnx_ops(args.onnx)
    write_json_result(result, args.out_json or None)
    return 0 if result["ok"] else 1


def check_tflite_ops_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)
    result = check_tflite_ops(args.tflite)
    write_json_result(result, args.out_json or None)
    return 0 if result["ok"] else 1


def compare_ethossafedet_reference_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--min-iou", type=float, default=0.85)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)
    result = compare_onnx_tflite_reference(args.onnx, args.tflite, args.image, input_size=args.input_size, min_iou=args.min_iou)
    write_json_result(result, args.out_json or None)
    return 0 if result["ok"] else 1


def run_host_mera_gate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--host-json", default="")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    parser.add_argument("--min-iou", type=float, default=0.85)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)
    result = run_host_mera_gate(args.reference_json, host_json=args.host_json or None, command=args.command, min_iou=args.min_iou)
    write_json_result(result, args.out_json or None)
    return 0 if result["ok"] else 1


def inspect_ruhmi_dispatch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--max-base-addr", type=int, default=8)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)
    result = inspect_ruhmi_dispatch(args.log, max_base_addr=args.max_base_addr)
    write_json_result(result, args.out_json or None)
    return 0 if result["ok"] else 1


def check_memory_budget_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="")
    parser.add_argument("--arena-bytes", type=int, default=None)
    parser.add_argument("--weights-bytes", type=int, default=None)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)
    result = check_memory_budget(args.arena_bytes, args.weights_bytes, log_path=args.log or None)
    write_json_result(result, args.out_json or None)
    return 0 if result["ok"] else 1


def make_static_golden_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        result = make_static_golden(args.onnx, args.manifest, args.out, input_size=args.input_size, limit=args.limit)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_result({"ok": True, "output_path": args.out, "item_count": len(result["items"])}, as_json=True)
    return 0


def make_ethossafedet_report_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a thesis-style EthosSafeDet-A formal chain report.")
    parser.add_argument("--run", required=True, help="Run directory containing train_report.json and gate artifacts.")
    parser.add_argument("--out", default="", help="Output Markdown path. Defaults to <run>/formal_chain_report.md.")
    parser.add_argument("--calibration", default="", help="Calibration manifest used for TFLite int8 export.")
    parser.add_argument("--reference-image", default="", help="Real camera image used by the PC ONNX vs TFLite reference gate.")
    args = parser.parse_args(argv)
    try:
        result = make_formal_chain_report(
            args.run,
            output_path=args.out or None,
            calibration_path=args.calibration or None,
            reference_image=args.reference_image or None,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_result({"ok": True, **result}, as_json=True)
    return 0


def _print_result(data: dict[str, Any], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(validate_self_dataset_main())
