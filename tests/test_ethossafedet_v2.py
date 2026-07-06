from __future__ import annotations

import json
import math
from pathlib import Path

import yaml
from PIL import Image

from assistive_grasp_detector.ethossafedet_v2_export import export_v2_onnx
from assistive_grasp_detector.ethossafedet_v2_gates import check_v2_onnx_ops, check_v2_weight_budget, write_gate
from assistive_grasp_detector.ethossafedet_v2_manifest import load_v2_manifest_records, prepare_ethossafedet_v2_manifest
from assistive_grasp_detector.ethossafedet_v2_manifest import stable_split_for_key
from assistive_grasp_detector.ethossafedet_v2_model import EthosSafeDetV2Config, V2_OUTPUT_NAMES, make_ethossafedet_v2, parameter_count
from assistive_grasp_detector.ethossafedet_v2_report import make_v2_formal_report
from assistive_grasp_detector.ethossafedet_v2_train import assign_v2_targets, train_ethossafedet_v2
from assistive_grasp_detector.schema import ETHOSSAFEDET_CLASS_NAMES


def test_v2_manifest_split_and_theta(tmp_path: Path) -> None:
    dataset = _make_v2_dataset(tmp_path / "dataset", images_per_class=3)
    manifest = tmp_path / "manifest.jsonl"
    result = prepare_ethossafedet_v2_manifest(dataset, manifest)
    assert result.ok, result.to_dict()
    records = load_v2_manifest_records(manifest)
    assert {record["split"] for record in records} == {"train", "val", "test"}
    theta_objects = [obj for record in records for obj in record["objects"] if obj["theta_valid"]]
    assert theta_objects
    assert all("orientation_sin2theta" in obj for obj in theta_objects)


def test_v2_targets_and_model_shapes() -> None:
    torch = __import__("torch")
    obj = {
        "class_id": 0,
        "bbox_xyxy_vga": [220.0, 160.0, 420.0, 320.0],
        "theta_valid": True,
        "orientation_sin2theta": 1.0,
        "orientation_cos2theta": 0.0,
    }
    targets = assign_v2_targets([obj], input_size=64, stride=8)
    assert targets["cls"].shape == (7, 8, 8)
    assert int(targets["positive"].sum()) > 0
    assert int(targets["orientation_mask"].sum()) > 0
    model = make_ethossafedet_v2(EthosSafeDetV2Config(input_size=64, width=16))
    outputs = model(torch.zeros((1, 3, 64, 64), dtype=torch.float32))
    assert len(outputs) == 6
    assert outputs[0].shape == (1, 7, 8, 8)
    assert outputs[1].shape == (1, 4, 8, 8)
    assert outputs[2].shape == (1, 2, 8, 8)
    assert outputs[3].shape == (1, 7, 4, 4)
    assert parameter_count(model) > 0


def test_v2_train_export_gate_and_report(tmp_path: Path) -> None:
    dataset = _make_v2_dataset(tmp_path / "dataset", images_per_class=3)
    manifest = tmp_path / "manifest.jsonl"
    result = prepare_ethossafedet_v2_manifest(dataset, manifest)
    assert result.ok, result.to_dict()
    run = tmp_path / "run"
    report = train_ethossafedet_v2(
        manifest,
        run,
        input_size=64,
        width=16,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        device="cpu",
        amp=False,
        cache_images=True,
        num_workers=0,
        eval_limit=4,
        min_int8_weight_bytes=1,
        max_int8_weight_bytes=1024 * 1024,
    )
    assert Path(report["checkpoint"]).is_file()
    onnx = run / "export" / "ethossafedet_v2_64.onnx"
    export_v2_onnx(report["checkpoint"], onnx, input_size=64)
    gates = run / "gates"
    write_gate(gates / "check_v2_onnx_ops.json", check_v2_onnx_ops(onnx, input_size=64))
    write_gate(gates / "check_v2_weight_budget.json", check_v2_weight_budget(run / "train_report.json", min_int8_bytes=1, max_int8_bytes=1024 * 1024))
    formal = make_v2_formal_report(run)
    assert Path(formal["report"]).is_file()
    assert formal["figure_count"] >= 8
    assert formal["table_count"] >= 6


def _make_v2_dataset(root: Path, images_per_class: int = 3) -> Path:
    img_dir = root / "images" / "camera_1"
    ann_dir = root / "annotations" / "camera_1"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    classes = {"classes": [{"id": i, "name": name, "graspable": True} for i, name in enumerate(ETHOSSAFEDET_CLASS_NAMES)]}
    (root / "classes.yaml").write_text(yaml.safe_dump(classes, sort_keys=False), encoding="utf-8")
    index = 0
    for class_id, name in enumerate(ETHOSSAFEDET_CLASS_NAMES):
        stems = _stems_for_all_splits(class_id)
        for repeat, stem in enumerate(stems):
            index += 1
            Image.new("RGB", (640, 480), color=(30 + class_id * 20, 40 + repeat * 20, 80)).save(img_dir / f"{stem}.png")
            x1 = 40.0 + class_id * 20.0
            y1 = 50.0 + repeat * 15.0
            obj = {
                "instance_id": 1,
                "class_id": class_id,
                "class_name": name,
                "bbox_xyxy": [x1, y1, x1 + 80.0, y1 + 90.0],
                "graspable": True,
                "yaw_label_status": "not_required",
            }
            if name in {"earbud", "phone", "remote", "tissue"}:
                obj["yaw_label_status"] = "valid"
                obj["main_axis_points"] = [[x1, y1], [x1 + 50.0, y1 + 20.0]]
            ann = {
                "image_id": stem,
                "image_path": str(img_dir / f"{stem}.png"),
                "width": 640,
                "height": 480,
                "split": "train",
                "objects": [obj],
            }
            (ann_dir / f"{stem}.json").write_text(json.dumps(ann, ensure_ascii=False), encoding="utf-8")
    return root


def _stems_for_all_splits(class_id: int) -> list[str]:
    found: dict[str, str] = {}
    candidate = class_id * 1000
    while set(found) != {"train", "val", "test"}:
        candidate += 1
        stem = f"{candidate:06d}"
        split = stable_split_for_key(f"camera_1/{stem}")
        found.setdefault(split, stem)
    return [found["train"], found["val"], found["test"]]
