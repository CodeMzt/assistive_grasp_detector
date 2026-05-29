# Experiment: Model A YOLOv8n PC Export Run 001

Date: 2026-05-29

## Goal

使用 `yolov8n.pt` COCO-pretrained detect model 跑通模型 A 的 PC 侧最小完整链路：

```text
yolov8n.pt
-> smoke training at imgsz=416
-> predict on one test image
-> export ONNX
-> export TFLite / INT8 candidate if dependencies allow
-> inspect output tensor and bbox mapping
```

本实验不做 RA8P1 上板，不验证 RUHMI，不引入模型 B。

## Environment

| Item | Value |
|---|---|
| Host OS | TBD |
| Python version | TBD |
| Ultralytics version | TBD |
| Torch version | TBD |
| ONNX version | TBD |
| TensorFlow / TFLite tooling | TBD |
| CUDA / device | TBD |

## Model

| Item | Value |
|---|---|
| Base model | `yolov8n.pt` |
| Pretraining dataset | COCO |
| Task | detect |
| Training image size | 416x416 |
| Final project classes | Not frozen in this run |

## Dataset

| Item | Value |
|---|---|
| Smoke dataset | `coco8.yaml` / TBD |
| Custom dataset | Not used in this run |
| Train images | TBD |
| Val images | TBD |
| Notes | This run validates the pipeline, not final accuracy. |

## Commands

Recommended local setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install ultralytics onnx onnxruntime tensorflow
```

Smoke training, predict and export:

```powershell
.\.venv\Scripts\python scripts\model_a_yolov8n_smoke.py `
  --data coco8.yaml `
  --weights yolov8n.pt `
  --imgsz 416 `
  --epochs 1 `
  --batch 2 `
  --device cpu `
  --source path\to\one_640x480_test_image.jpg `
  --export-onnx `
  --export-tflite
```

Coordinate contract demo:

```powershell
python scripts\model_a_letterbox_demo.py
```

## Expected Outputs

| Artifact | Expected path / note |
|---|---|
| Training run | `runs/model_a_yolov8n_pc_export_run_001/train` |
| Predict run | `runs/model_a_yolov8n_pc_export_run_001/predict` |
| Best checkpoint | ignored by git |
| ONNX export | ignored by git, record path/hash here |
| TFLite export | ignored by git, record path/hash here |
| INT8 export | optional; record failure if dependencies/calibration fail |

## Observed Output

```text
TBD
```

## Export Results

| Format | Result | Artifact | Notes |
|---|---|---|---|
| PyTorch `.pt` | TBD | TBD | Baseline output |
| ONNX | TBD | TBD | `nms=False`; raw detector output preferred |
| TFLite FP32/FP16 | TBD | TBD | Candidate only |
| TFLite INT8 | TBD | TBD | Candidate only; may need calibration data |

## Output Tensor Notes

| Item | Value |
|---|---|
| Raw tensor shape | TBD |
| Score computation | TBD |
| NMS parameters | TBD |
| Confidence threshold | TBD |
| IoU threshold | TBD |

## Single Image Prediction

| Item | Value |
|---|---|
| Source image | TBD |
| Source resolution | 640x480 expected |
| Top detection class | TBD |
| Top detection confidence | TBD |
| Top detection bbox in source coords | TBD |

Example normalized external result:

```c
semantic_det_raw_t det = {
    .class_id = 0,
    .confidence = 0.0f,
    .bbox_x1 = 0.0f,
    .bbox_y1 = 0.0f,
    .bbox_x2 = 0.0f,
    .bbox_y2 = 0.0f,
};
```

## Coordinate Mapping Check

| Item | Value |
|---|---|
| VGA input | 640x480 |
| Model A input | 416x416 letterbox |
| Letterbox scale | TBD |
| Letterbox pad x/y | TBD |
| bbox in 416 coords | TBD |
| bbox mapped back to VGA | TBD |
| Visual check | TBD |

## Result

TBD

## Problems

TBD

## Next Step

TBD
