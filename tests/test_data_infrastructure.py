from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from assistive_grasp_detector.annotator_dataset import (
    stable_split_for_key,
    validate_self_dataset,
)
from assistive_grasp_detector.coco_subset import prepare_coco_subset
from assistive_grasp_detector.model_b_index import index_model_b_targets
from assistive_grasp_detector.yolo_export import build_model_a_yolo


def test_self_dataset_validation_and_yolo_export(tmp_path: Path) -> None:
    dataset = _make_self_dataset(tmp_path / "self_dataset")

    report = validate_self_dataset(dataset)

    assert report.ok
    assert report.image_count == 2
    assert report.object_count == 2
    assert any(issue.code == "split_missing" for issue in report.warnings)

    out = tmp_path / "generated" / "model_a" / "self_v0"
    result = build_model_a_yolo(dataset, out)

    assert result.label_count == 2
    label = (out / "labels" / "train" / "board_vga" / "000001.txt").read_text(encoding="utf-8").strip()
    assert label == "0 0.500000 0.500000 0.250000 0.250000"

    fallback_split = stable_split_for_key("board_vga/000002.jpg")
    assert (out / "labels" / fallback_split / "board_vga" / "000002.txt").is_file()
    dataset_yaml = yaml.safe_load((out / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["names"][0] == "phone_A"
    assert dataset_yaml["names"][6] == "cup_other"


def test_coco_subset_mapping_cap_and_exclusion(tmp_path: Path) -> None:
    classes_path = _write_classes(tmp_path / "classes.yaml")
    coco_root = tmp_path / "coco2017"
    _make_coco_fixture(coco_root)
    config = tmp_path / "coco_config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "target_classes": str(classes_path),
                "splits": {
                    "train": {
                        "images": "train2017",
                        "annotations": "annotations/instances_train2017.json",
                    }
                },
                "max_per_class": {"train": 1},
                "category_map": {
                    "bottle": "bottle_other",
                    "cup": "cup_other",
                    "book": "book",
                    "cell phone": "phone_other",
                },
                "exclude_categories": ["remote"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = tmp_path / "generated" / "model_a" / "coco_subset_v0"
    result = prepare_coco_subset(coco_root, config, out)

    assert result.ok
    assert result.split_counts["train"] == 2
    assert result.label_counts["train"] == 2
    label_files = sorted((out / "labels" / "train").glob("*.txt"))
    labels = "\n".join(path.read_text(encoding="utf-8") for path in label_files)
    label_lines = [line for line in labels.splitlines() if line.strip()]
    assert len(label_lines) == 2
    assert "8 " in labels
    assert "6 " in labels
    assert "1 " not in labels
    assert not (out / "images" / "train" / "000000000003.jpg").exists()


def test_model_b_target_index_validation(tmp_path: Path) -> None:
    root = tmp_path / "target_maps"
    item_dir = root / "000001"
    item_dir.mkdir(parents=True)
    q_map = np.zeros((4, 4), dtype=np.float32)
    q_map[1, 1] = 1.0
    np.savez_compressed(
        item_dir / "obj_001.npz",
        q_map=q_map,
        sin2theta_map=np.zeros((4, 4), dtype=np.float32),
        cos2theta_map=np.ones((4, 4), dtype=np.float32),
        width_map=np.full((4, 4), 0.25, dtype=np.float32),
    )
    Image.new("RGB", (4, 4)).save(item_dir / "obj_001.png")
    (item_dir / "obj_001.json").write_text(
        json.dumps(
            {
                "source_image": "images/board_vga/000001.jpg",
                "source_bbox": [1, 2, 3, 4],
                "padded_bbox": [0, 1, 4, 5],
                "map_size": 4,
                "instance_id": 1,
                "class_id": 0,
                "class_name": "phone_A",
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "generated" / "manifests" / "model_b_self_v0.jsonl"
    result = index_model_b_targets(root, out)

    assert result.ok
    assert result.record_count == 1
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["target_npz"] == "000001/obj_001.npz"
    assert record["positive_pixels"] == 1
    assert record["map_size"] == 4


def _make_self_dataset(root: Path) -> Path:
    _write_classes(root / "classes.yaml")
    image_dir = root / "images" / "board_vga"
    ann_dir = root / "annotations" / "board_vga"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    Image.new("RGB", (640, 480), color=(32, 32, 32)).save(image_dir / "000001.jpg")
    Image.new("RGB", (640, 480), color=(64, 64, 64)).save(image_dir / "000002.jpg")
    _write_annotation(
        ann_dir / "000001.json",
        "000001",
        "images/board_vga/000001.jpg",
        "train",
        class_id=0,
        class_name="phone_A",
        bbox=[240, 180, 400, 300],
    )
    _write_annotation(
        ann_dir / "000002.json",
        "000002",
        "images/board_vga/000002.jpg",
        None,
        class_id=6,
        class_name="cup_other",
        bbox=[100, 100, 200, 220],
    )
    return root


def _write_classes(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "classes": [
                    {"id": 0, "name": "phone_A", "graspable": True, "policy": "grasp_rect"},
                    {"id": 6, "name": "cup_other", "graspable": False, "policy": "report_only"},
                    {"id": 7, "name": "phone_other", "graspable": False, "policy": "report_only"},
                    {"id": 8, "name": "bottle_other", "graspable": False, "policy": "report_only"},
                    {"id": 9, "name": "book", "graspable": False, "policy": "report_only"},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_annotation(
    path: Path,
    image_id: str,
    image_path: str,
    split: str | None,
    class_id: int,
    class_name: str,
    bbox: list[int],
) -> None:
    data = {
        "image_id": image_id,
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
                "graspable": class_id == 0,
                "policy": "grasp_rect" if class_id == 0 else "report_only",
                "grasps": [],
            }
        ],
    }
    if split is not None:
        data["split"] = split
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_coco_fixture(root: Path) -> None:
    ann_dir = root / "annotations"
    img_dir = root / "train2017"
    ann_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    for index in range(1, 4):
        Image.new("RGB", (100, 100), color=(index, index, index)).save(img_dir / f"{index:012d}.jpg")

    coco = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "000000000002.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "000000000003.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 44, "name": "bottle"},
            {"id": 47, "name": "cup"},
            {"id": 65, "name": "remote"},
            {"id": 1, "name": "person"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 44, "bbox": [10, 10, 20, 30]},
            {"id": 2, "image_id": 2, "category_id": 44, "bbox": [10, 10, 20, 30]},
            {"id": 3, "image_id": 2, "category_id": 47, "bbox": [20, 20, 10, 10]},
            {"id": 4, "image_id": 3, "category_id": 65, "bbox": [5, 5, 10, 10]},
            {"id": 5, "image_id": 3, "category_id": 1, "bbox": [0, 0, 50, 50]},
        ],
    }
    (ann_dir / "instances_train2017.json").write_text(json.dumps(coco), encoding="utf-8")
