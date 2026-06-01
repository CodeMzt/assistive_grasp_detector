"""Command line entry points for data infrastructure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from assistive_grasp_detector.annotator_dataset import validate_self_dataset
from assistive_grasp_detector.coco_subset import prepare_coco_subset
from assistive_grasp_detector.model_b_index import index_model_b_targets
from assistive_grasp_detector.yolo_export import build_model_a_yolo


def validate_self_dataset_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an AssistiveGraspAnnotator dataset.")
    parser.add_argument("--dataset", required=True, help="Path to the annotator dataset root.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    report = validate_self_dataset(args.dataset)
    _print_result(report.to_dict(), as_json=args.json)
    return 0 if report.ok else 1


def build_model_a_yolo_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build YOLO data for Model A from self-collected annotations.")
    parser.add_argument("--dataset", required=True, help="Path to the annotator dataset root.")
    parser.add_argument("--out", required=True, help="Output directory under data/generated.")
    parser.add_argument("--classes", default=None, help="Optional classes.yaml override.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    try:
        result = build_model_a_yolo(args.dataset, args.out, args.classes)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_result(result.to_dict(), as_json=args.json)
    return 0


def index_model_b_targets_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index Model B target maps exported by the annotator.")
    parser.add_argument("--target-maps", required=True, help="Path to generated/target_maps.")
    parser.add_argument("--out", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    result = index_model_b_targets(args.target_maps, args.out)
    _print_result(result.to_dict(), as_json=args.json)
    return 0 if result.ok else 1


def prepare_coco_subset_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a local COCO subset for Model A.")
    parser.add_argument("--coco-root", required=True, help="Path to local COCO 2017 root.")
    parser.add_argument("--config", required=True, help="COCO subset config YAML.")
    parser.add_argument("--out", required=True, help="Output directory under data/generated.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    try:
        result = prepare_coco_subset(args.coco_root, args.config, args.out)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_result(result.to_dict(), as_json=args.json)
    return 0 if result.ok else 1


def _print_result(data: dict[str, Any], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(validate_self_dataset_main())
