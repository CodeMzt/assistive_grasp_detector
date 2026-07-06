from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from assistive_grasp_detector.annotator_dataset import validate_self_dataset
from assistive_grasp_detector.ethossafedet_manifest import (
    build_calibration_manifest,
    load_manifest_records,
    prepare_ethossafedet_manifest_from_export,
    prepare_ethossafedet_manifest_from_self_dataset,
)
from assistive_grasp_detector.ethossafedet_export import export_onnx_reference
from assistive_grasp_detector.ethossafedet_train import assign_targets, train_ethossafedet_a


def test_self_dataset_validation_and_ethossafedet_manifest(tmp_path: Path) -> None:
    dataset = _make_self_dataset(tmp_path / "self_dataset")
    report = validate_self_dataset(dataset)

    assert report.ok
    assert report.image_count == 2
    assert report.object_count == 2
    assert any(issue.code == "split_missing" for issue in report.warnings)

    out = tmp_path / "ethossafedet_manifest.jsonl"
    result = prepare_ethossafedet_manifest_from_self_dataset(dataset, out)

    assert result.ok
    assert result.record_count == 2
    assert result.object_count == 2
    records = load_manifest_records(out)
    first = records[0]
    assert first["schema_version"] == "ethossafedet_manifest_v1"
    assert first["width"] == 640
    assert first["height"] == 480
    assert first["objects"][0]["class_name"] == "earbud"
    assert first["objects"][0]["bbox_xyxy_vga"] == [240.0, 180.0, 400.0, 300.0]


def test_export_layout_manifest_conversion_with_negative_whitelist(tmp_path: Path) -> None:
    dataset = _make_export_dataset(tmp_path / "export_dataset", count=4)
    out = tmp_path / "ethossafedet_manifest.jsonl"

    result = prepare_ethossafedet_manifest_from_export(dataset, out, negative_image_ids=["000003"])

    assert result.ok
    assert result.record_count == 4
    assert result.object_count == 3
    assert result.negative_count == 1
    assert result.excluded_count == 0
    records = load_manifest_records(out)
    positive = next(record for record in records if record["image"].endswith("000001.png"))
    assert positive["objects"][0]["class_id"] == 0
    assert positive["objects"][0]["bbox_xyxy_vga"] == [240.0, 180.0, 400.0, 300.0]
    negative = next(record for record in records if record["image"].endswith("000003.png"))
    assert negative["negative"] is True
    assert negative["objects"] == []


def test_calibration_manifest_rejects_less_than_200_images(tmp_path: Path) -> None:
    dataset = _make_export_dataset(tmp_path / "export_dataset", count=3)
    manifest = tmp_path / "ethossafedet_manifest.jsonl"
    prepare_ethossafedet_manifest_from_export(dataset, manifest, negative_image_ids=["000003"])

    with pytest.raises(ValueError, match="need 200 real VGA camera images"):
        build_calibration_manifest(manifest, tmp_path / "calibration.json", target_count=200)


def test_calibration_manifest_accepts_200_real_images(tmp_path: Path) -> None:
    dataset = _make_export_dataset(tmp_path / "export_dataset", count=200)
    manifest = tmp_path / "ethossafedet_manifest.jsonl"
    result = prepare_ethossafedet_manifest_from_export(dataset, manifest, negative_image_ids=["000003"])
    assert result.ok

    calibration = build_calibration_manifest(manifest, tmp_path / "calibration.json", target_count=200, seed=123)

    assert calibration["schema_version"] == "ethossafedet_calibration_v1"
    assert len(calibration["items"]) == 200
    assert all(Path(item["image"]).is_file() for item in calibration["items"])


def test_assign_targets_ltrb_center_cell() -> None:
    targets = assign_targets(
        [{"class_id": 2, "bbox_xyxy_vga": [240.0, 180.0, 400.0, 300.0]}],
        input_size=320,
        stride=8,
    )

    assert targets["positive"].sum() >= 1
    gy, gx = np.argwhere(targets["positive"])[0]
    assert targets["cls"][2, gy, gx] > 0.0
    assert targets["cls"][2, gy, gx] <= 1.0
    assert np.all(targets["box"][:, gy, gx] > 0)


def test_train_report_outputs_json_csv_markdown_and_best_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    dataset = _make_train_report_dataset(tmp_path / "train_report_dataset")
    manifest = tmp_path / "ethossafedet_manifest.jsonl"
    result = prepare_ethossafedet_manifest_from_self_dataset(dataset, manifest)
    assert result.ok

    run_dir = tmp_path / "run"
    report = train_ethossafedet_a(
        manifest,
        run_dir,
        input_size=32,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        seed=7,
        eval_score_threshold=0.25,
        nms_iou_threshold=0.5,
    )

    assert report["schema_version"] == "ethossafedet_train_report_v1"
    assert report["best_epoch"] == 1
    assert Path(report["checkpoint"]).is_file()
    assert Path(report["last_checkpoint"]).is_file()
    assert report["checkpoint_sha256"]
    assert report["last_checkpoint_sha256"]
    assert len(report["history"]) == 1
    assert report["history"][0]["val_top1_class_acc"] is not None
    report_paths = report["report_paths"]
    assert Path(report_paths["json"]).is_file()
    assert Path(report_paths["csv"]).is_file()
    assert Path(report_paths["markdown"]).is_file()

    saved_report = json.loads(Path(report_paths["json"]).read_text(encoding="utf-8"))
    assert saved_report["data"]["record_count"] == 4
    assert saved_report["hyperparameters"]["seed"] == 7
    csv_lines = Path(report_paths["csv"]).read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("epoch,train_loss,train_cls_loss")
    assert len(csv_lines) == 2
    assert "EthosSafeDet-A Training Report" in Path(report_paths["markdown"]).read_text(encoding="utf-8")

    onnx_path = run_dir / "ethossafedet_a_32.onnx"
    exported = export_onnx_reference(report["checkpoint"], onnx_path, input_size=32)
    assert Path(exported["onnx"]).is_file()


def _make_self_dataset(root: Path) -> Path:
    _write_ethos_classes(root / "classes.yaml")
    image_dir = root / "images" / "board_vga"
    ann_dir = root / "annotations" / "board_vga"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.new("RGB", (640, 480), color=(32, 32, 32)).save(image_dir / "000001.jpg")
    Image.new("RGB", (640, 480), color=(64, 64, 64)).save(image_dir / "000002.jpg")
    _write_annotation(ann_dir / "000001.json", "images/board_vga/000001.jpg", "train", 0, "earbud", [240, 180, 400, 300])
    _write_annotation(ann_dir / "000002.json", "images/board_vga/000002.jpg", None, 1, "phial", [100, 100, 200, 220])
    return root


def _make_train_report_dataset(root: Path) -> Path:
    _write_ethos_classes(root / "classes.yaml")
    image_dir = root / "images" / "board_vga"
    ann_dir = root / "annotations" / "board_vga"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    rows = [
        ("000001", "train", 0, "earbud", [240, 180, 400, 300], (32, 32, 32)),
        ("000002", "train", 1, "phial", [100, 100, 220, 240], (64, 64, 64)),
        ("000003", "val", 2, "bottle", [200, 160, 360, 320], (96, 96, 96)),
        ("000004", "val", 3, "phone", [260, 170, 430, 310], (128, 128, 128)),
    ]
    for stem, split, class_id, class_name, bbox, color in rows:
        Image.new("RGB", (640, 480), color=color).save(image_dir / f"{stem}.jpg")
        _write_annotation(ann_dir / f"{stem}.json", f"images/board_vga/{stem}.jpg", split, class_id, class_name, bbox)
    return root


def _make_export_dataset(root: Path, count: int) -> Path:
    (root / "images" / "camera_1").mkdir(parents=True)
    (root / "camera_1").mkdir(parents=True)
    _write_ethos_classes(root / "classes.yaml")
    for index in range(1, count + 1):
        stem = f"{index:06d}"
        Image.new("RGB", (640, 480), color=(index % 255, 16, 16)).save(root / "images" / "camera_1" / f"{stem}.png")
        if stem == "000003":
            continue
        class_id = (index - 1) % 7
        (root / "camera_1" / f"{stem}.txt").write_text(
            f"{class_id} 0.500000 0.500000 0.250000 0.250000\n",
            encoding="utf-8",
        )
    return root


def _write_ethos_classes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["earbud", "phial", "bottle", "phone", "remote", "tissue", "apple"]
    path.write_text(
        yaml.safe_dump(
            {"classes": [{"id": i, "name": name, "graspable": True, "policy": "bbox"} for i, name in enumerate(names)]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_annotation(
    path: Path,
    image_path: str,
    split: str | None,
    class_id: int,
    class_name: str,
    bbox: list[int],
) -> None:
    data = {
        "image_id": path.stem,
        "image_path": image_path,
        "width": 640,
        "height": 480,
        "camera": "board_vga",
        "source": "board",
        "objects": [
            {
                "instance_id": 1,
                "class_id": class_id,
                "class_name": class_name,
                "bbox_xyxy": bbox,
                "graspable": True,
                "policy": "bbox",
                "grasps": [],
            }
        ],
    }
    if split is not None:
        data["split"] = split
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
