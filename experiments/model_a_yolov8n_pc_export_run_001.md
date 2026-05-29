# Experiment: Model A YOLOv8n PC Export Run 001

Date: 2026-05-29

## Goal

使用 `yolov8n.pt` COCO-pretrained detect model 跑通模型 A 的 PC 侧最小完整链路：

```text
yolov8n.pt
-> smoke training at imgsz=416
-> predict on one test image
-> export ONNX
-> inspect output tensor and bbox mapping
```

本实验不做 RA8P1 上板，不验证 e2 studio 转换/量化，不引入模型 B。训练侧交付物限定为 `.pt` 和 `.onnx`。

## Environment

| Item | Value |
|---|---|
| Host OS | Windows |
| Conda env | `cv_detection` |
| Python version | 3.10.19 |
| Ultralytics version | 8.4.47 |
| Torch version | 2.9.1+cu128 |
| ONNX version | 1.20.1 |
| ONNX Runtime | 1.23.2 |
| CUDA / device | CUDA available, NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB |

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
| Smoke dataset | `coco8.yaml` |
| Custom dataset | Not used in this run |
| Train images | 4 |
| Val images | 4 |
| Notes | This run validates the pipeline, not final accuracy. |

## Commands

Actual environment used:

```powershell
D:\anaconda3\envs\cv_detection\python.exe
```

Smoke training, predict and export:

```powershell
& 'D:\anaconda3\envs\cv_detection\python.exe' scripts\model_a_yolov8n_smoke.py `
  --data coco8.yaml `
  --weights yolov8n.pt `
  --imgsz 416 `
  --epochs 1 `
  --batch 2 `
  --device 0 `
  --source captures\model_a_test_640x480.jpg `
  --export-onnx `
  --export-nms-onnx
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
| NMS ONNX export | optional PC reference, ignored by git |

## Observed Output

```text
1 epoch smoke training completed on CUDA.

Validation on coco8:
precision(B)=0.59262
recall(B)=0.73333
mAP50(B)=0.82206
mAP50-95(B)=0.61488

Single image predict on captures/model_a_test_640x480.jpg:
4 persons, 1 bus
```

## Export Results

| Format | Result | Artifact | Notes |
|---|---|---|---|
| PyTorch `.pt` | PASS | `runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best.pt` | SHA256 `54B478C1F693BAFB694E3FA203FF55C416137FD4D6B1E6602EDE8427D5262583` |
| raw ONNX | PASS | `runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best_raw.onnx` | SHA256 `F0B648B9FA6E6950FCA0B097D59F8C6B9D89A04AA37A00DD24088C59BA982340`; board-side preferred candidate |
| NMS ONNX | PASS | `runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best_nms.onnx` | SHA256 `A0DEBB0BA4D85143199F07B66278489B0F392BC7832D189B9DC798C940E196E8`; PC reference only |

## Output Tensor Notes

| Item | Value |
|---|---|
| Raw tensor shape | ONNX input `images [1,3,416,416]`, output `output0 [1,84,3549]` |
| NMS ONNX shape | ONNX input `images [1,3,416,416]`, output `output0 [1,300,6]` |
| Score computation | YOLOv8 detect head, class score from output channels after box channels |
| NMS parameters | PC predict used Ultralytics default flow with `conf=0.25`, `iou=0.7` |
| Confidence threshold | 0.25 |
| IoU threshold | 0.7 |

## Single Image Prediction

| Item | Value |
|---|---|
| Source image | `captures/model_a_test_640x480.jpg` |
| Source resolution | 640x480 |
| Top detection class | `bus`, COCO class_id `5` |
| Top detection confidence | 0.8668924570083618 |
| Top detection bbox in source coords | `[10.5671, 102.7513, 638.1050, 335.4526]` |

Example normalized external result:

```c
semantic_det_raw_t det = {
    .class_id = 5,
    .confidence = 0.86689246f,
    .bbox_x1 = 10.5671f,
    .bbox_y1 = 102.7513f,
    .bbox_x2 = 638.1050f,
    .bbox_y2 = 335.4526f,
};
```

## Coordinate Mapping Check

| Item | Value |
|---|---|
| VGA input | 640x480 |
| Model A input | 416x416 letterbox |
| Letterbox scale | 0.65 |
| Letterbox pad x/y | `pad_x=0`, `pad_y=52` |
| bbox in 416 coords | Demo box `[130,117,286,273]` |
| bbox mapped back to VGA | `[200,100,440,340]` |
| Visual check | Coordinate demo script passed |

## Result

PASS for training-side PT/ONNX deliverable.

The valid handoff artifacts for the board-side computer are:

```text
runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best.pt
runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best_raw.onnx
runs/detect/runs/model_a_yolov8n_pc_export_run_001/train/weights/best_nms.onnx
```

Board-side e2 studio / RA tools will handle conversion and quantization.

## Problems

1. The generated run path contains an extra `runs/detect/runs/...` segment because of Ultralytics project path handling in the first script revision. This does not affect the artifacts.
2. Raw ONNX is the board-side preferred candidate, but PC-side Ultralytics direct predict from raw ONNX produced abnormal bbox scaling. NMS ONNX and PyTorch predict produced sane boxes. Treat raw ONNX as model-tensor handoff, not as a completed PC post-processing proof.
3. Quantization/conversion is intentionally left to the board-side e2 studio / RA toolchain.

## Next Step

1. Use `best_raw.onnx` first on the board-side e2 studio conversion/quantization toolchain.
2. If the board-side tool has trouble with raw YOLO output/post-processing, try `best_nms.onnx` as a diagnostic reference, but do not assume NMS ONNX is the final embedded path.
3. Replace `coco8.yaml` with the first self-collected tabletop dataset when available.
