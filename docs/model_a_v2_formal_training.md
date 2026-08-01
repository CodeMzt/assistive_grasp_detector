# Model A V2 Formal Training

This document records the detector-side formal training path for Model A V2 /
EthosSafeDetV2. It is a host-side training and reporting workflow, not a board
acceptance record.

## Dataset

The current formal dataset source is:

```text
D:\AssistiveGraspAnnotatorData\datasets\new_dataset
```

The V2 manifest builder reads scene-level annotations and writes a JSONL
manifest with canonical seven-class IDs, VGA `bbox_xyxy_vga`, a deterministic
80/10/10 train/validation/test split, and optional orientation supervision
through `theta_valid` plus `sin(2theta), cos(2theta)`.

The split is computed from the annotation-relative image key and does not modify
the source annotation JSON files.

## Training

Use the ma2 PyTorch environment, not the detector `.venv`:

```powershell
$env:PYTHONPATH='D:\Project\assistive_grasp_detector'
& 'D:\Anaconda3\envs\env_isaaclab\python.exe' -m assistive_grasp_detector.ethossafedet_v2_manifest `
  --dataset 'D:\AssistiveGraspAnnotatorData\datasets\new_dataset' `
  --out 'D:\Project\assistive_grasp_detector\runs\model_a_v2_w40_320_20260706_formal_r2\manifest\ethossafedet_v2_manifest.jsonl' `
  --json

& 'D:\Anaconda3\envs\env_isaaclab\python.exe' -m assistive_grasp_detector.ethossafedet_v2_train `
  --manifest 'D:\Project\assistive_grasp_detector\runs\model_a_v2_w40_320_20260706_formal_r2\manifest\ethossafedet_v2_manifest.jsonl' `
  --out 'D:\Project\assistive_grasp_detector\runs\model_a_v2_w40_320_20260706_formal_r2' `
  --input-size 320 --width 40 --epochs 120 --batch 24 --lr 0.0003 `
  --weight-decay 0.0001 --device cuda --seed 0 --eval-every 1 --num-workers 0 --amp
```

The model emits six separated tensors:

```text
s8_cls_logits, s8_box_ltrb, s8_orientation,
s16_cls_logits, s16_box_ltrb, s16_orientation
```

The trainer also writes `ethossafedet_v2_best_live.pt` and
`ethossafedet_v2_last_live.pt` after each epoch so a long ma2 run can be
recovered if the host process fails before final report generation.

## Formal Result

Completed run:

```text
D:\Project\assistive_grasp_detector\runs\model_a_v2_w40_320_20260706_formal_r2
```

Dataset manifest summary:

| Split | Images |
| --- | ---: |
| train | 2621 |
| val | 316 |
| test | 303 |

| Class | Objects |
| --- | ---: |
| bottle | 990 |
| earbud | 1116 |
| phial | 699 |
| phone | 1322 |
| remote | 1264 |
| tissue | 501 |
| apple | 794 |

Architecture and size:

| Item | Value |
| --- | ---: |
| input | 320x320 |
| width | 40 |
| classes | 7 |
| parameters | 239226 |
| estimated INT8 weight bytes | 239226 |
| estimated FP32 weight bytes | 956904 |

Validation best checkpoint:

| Metric | Value |
| --- | ---: |
| best epoch | 115 |
| recall@0.5 | 0.9753 |
| mean best IoU | 0.8352 |
| theta MAE rad | 0.0736 |

Held-out test metrics:

| Metric | Value |
| --- | ---: |
| GT objects | 629 |
| recall@0.5 | 0.9603 |
| mean best IoU | 0.8209 |
| theta MAE rad | 0.0681 |

Static gates:

| Gate | Result |
| --- | --- |
| ONNX op set | pass (`Add`, `Conv`, `Relu`) |
| INT8 weight budget | pass (239226 bytes) |

## Report

After training and ONNX static gates, generate the paper-style report:

```powershell
& 'D:\Anaconda3\envs\env_isaaclab\python.exe' -m assistive_grasp_detector.ethossafedet_v2_report `
  --run 'D:\Project\assistive_grasp_detector\runs\model_a_v2_w40_320_20260706_formal_r2'
```

The report writes `formal_report.md` plus `formal_report_assets/*.png` and
`*.csv` files under the run directory. The completed formal report contains
12 PNG figures, including validation qualitative success and hard-case panels,
plus CSV tables for split distribution, class distribution, theta coverage,
class weights, epoch history, qualitative validation examples, per-class
validation/test metrics, gate status, and artifact provenance.

## Acceptance Boundary

This workflow can produce a host-side formal training candidate and ONNX static
gate evidence. It does not prove TFLite full-int8, host MERA, RUHMI dispatch, or
board static golden. Those remain separate acceptance gates before firmware
consumes the candidate.

## R3 Real-Scene Retraining

R3 keeps the completed R2 manifest membership frozen and adds a separate,
terminal real-scene holdout. It records the current tissue variant under the
existing `tissue` class, gives new real captures containing `phial`, `bottle`,
or `phone` a bounded `1.5x` sampler multiplier, and gives six training
empty-table images a `2.0x` multiplier. The combined sampler weight remains
capped at `3.0`; apple and earbud are never downsampled.

The R3 policy owns the nine empty-table frames `camera_1/003347` through
`003355`, excludes `004056`, `004057`, `004138`, and `004170`, and uses the
new real-capture range `003870..004545`. Existing R2 train/validation/test
membership is copied exactly. New real captures use a deterministic `80/10/10`
train/validation/real-scene-holdout allocation; the holdout is evaluated only
after the validation-selected checkpoint has been fixed.

```powershell
$env:PYTHONPATH='D:\Project\assistive_grasp_detector'
$run = 'D:\Project\assistive_grasp_detector\runs\model_a_v2_w40_320_20260715_realdata_r3'
$r2Manifest = 'D:\Project\assistive_grasp_detector\runs\model_a_v2_w40_320_20260706_formal_r2\manifest\ethossafedet_v2_manifest.jsonl'

& 'D:\Anaconda3\envs\env_isaaclab\python.exe' -m assistive_grasp_detector.ethossafedet_v2_r3 prepare `
  --dataset 'D:\AssistiveGraspAnnotatorData\datasets\new_dataset' `
  --r2-manifest $r2Manifest `
  --out "$run\manifest\ethossafedet_v2_r3_manifest.jsonl" `
  --write-empty-annotations

& 'D:\Anaconda3\envs\env_isaaclab\python.exe' -m assistive_grasp_detector.ethossafedet_v2_train `
  --manifest "$run\manifest\ethossafedet_v2_r3_manifest.jsonl" `
  --out $run `
  --input-size 320 --width 40 --epochs 120 --batch 24 --lr 0.0003 `
  --weight-decay 0.0001 --device cuda --seed 0 --eval-every 1 --num-workers 0 --amp

& 'D:\Anaconda3\envs\env_isaaclab\python.exe' -m assistive_grasp_detector.ethossafedet_v2_r3 evaluate `
  --run $run `
  --r2-run 'D:\Project\assistive_grasp_detector\runs\model_a_v2_w40_320_20260706_formal_r2'

# Run the unchanged ONNX export and static gates, then extend the paper-style report.
& 'D:\Anaconda3\envs\env_isaaclab\python.exe' -m assistive_grasp_detector.ethossafedet_v2_r3 report --run $run
```

The R3 report preserves the R2 frozen-test comparison, adds real-scene and
empty-table terminal metrics at the fixed `0.25` score threshold, and renders
real-scene plus empty-table qualitative panels. This is still a host-side
candidate report, not board acceptance evidence.
