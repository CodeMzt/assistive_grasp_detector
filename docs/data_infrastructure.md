# Data Infrastructure V2 Contract

当前系统数据合同与固件 Model A V2 对齐：从 OV5640 VGA 640x480 场景级母图与 bbox/class/orientation 标注生成中性的 `ethossafedet_manifest.jsonl`，再用于训练、校准、导出和 gate。旧 bbox-only manifest 与 target-map/ROI 工具只能作为迁移输入或 deprecated reference。

## Board-Domain Source

推荐源数据形态：

```text
dataset_root/
  classes.yaml
  images/
    board_vga/
      000001.jpg
  annotations/
    board_vga/
      000001.json
```

现有首批导出数据仍可作为迁移输入：

```text
export_dataset/
  classes.yaml
  images/
    camera_1/
      000001.png
  camera_1/
    000001.txt
```

其中 `camera_1/*.txt` 只作为 legacy normalized bbox source 读取，格式为 `class_id cx cy w h`；它缺少 orientation，不满足 Model A V2 训练合同。生成后的主契约必须包含 `bbox_xyxy_vga`，并在有方向目标上包含 `theta_valid=true` 与 orientation 标签。

## Manifest Contract

`ethossafedet_manifest.jsonl` 每行是一张 640x480 图：

```json
{
  "schema_version": "ethossafedet_manifest_v2",
  "dataset_root": "D:/Project/assistive_grasp_detector/data/raw/model_a/first_batch_20260604",
  "image": "images/camera_1/000001.png",
  "split": "train",
  "width": 640,
  "height": 480,
  "negative": false,
  "objects": [
    {
      "class_id": 0,
      "class_name": "earbud",
      "bbox_xyxy_vga": [395.0, 137.0, 482.0, 219.0],
      "theta_valid": true,
      "grasp_yaw": 0.42,
      "orientation_sin2theta": 0.744643,
      "orientation_cos2theta": 0.667462
    }
  ]
}
```

V2 类别必须严格等于：

```text
0 earbud
1 phial
2 bottle
3 phone
4 remote
5 tissue
```

## Commands

```powershell
validate_self_dataset --dataset D:\path\to\dataset

prepare_ethossafedet_manifest `
  --source-format self `
  --dataset D:\path\to\dataset `
  --out data\generated\ethossafedet_a\self_v1\ethossafedet_manifest.jsonl

prepare_ethossafedet_manifest `
  --source-format export `
  --dataset data\raw\model_a\first_batch_20260604 `
  --out data\generated\ethossafedet_a\first_batch_20260604\ethossafedet_manifest.jsonl `
  --negative-image-id 000139

build_ethossafedet_calibration `
  --manifest data\generated\ethossafedet_a\first_batch_20260604\ethossafedet_manifest.jsonl `
  --out data\generated\ethossafedet_a\first_batch_20260604\calibration_320.json `
  --target-count 320
```

## Orientation Rules

- Orientation is image-plane theta in the VGA mother image, encoded as `sin(2theta), cos(2theta)` for model training and raw output.
- `theta_valid=true` is required for graspable preset objects with a stable main axis.
- Directionless, symmetric, ambiguous, or occluded objects must keep `theta_valid=false`; they may still contribute class/bbox supervision.
- Legacy bbox-only records can be used for class/bbox migration, but they cannot be counted as V2 orientation coverage.

## Calibration Rules

- Calibration source must be an EthosSafeDet manifest with real 640x480 board-camera images.
- `target_count` must be in `[200, 500]`.
- Every selected image path is written explicitly to `ethossafedet_calibration_v1` JSON.
- Random arrays, temporary screenshots, and single-image smoke inputs are rejected by construction.

## Notes

Legacy target-map/ROI indexing remains available as deprecated reference tooling only. It is not part of Model A V2 training, firmware deployment, or final acceptance.

## 2026-06-14 object_vocab_v1 alignment

Detector training and annotation configs now consume `configs/classes/object_vocab_v1.json` and the canonical seven-class order `0 earbud`, `1 phial`, `2 bottle`, `3 phone`, `4 remote`, `5 tissue`, `6 apple`. Existing six-class board exports remain legacy until a retrained seven-class model is exported and accepted by main firmware static golden checks.

