# Experiment: EthosSafeDet-A v1 Gate Record

Date: YYYY-MM-DD

## Scope

记录一次 EthosSafeDet-A v1 从训练/导出到 host MERA/RUHMI/static golden 的 gate 结果。只陈述实际命令、产物、日志和测量值，不把未实测内容写成事实。

## Environment

| Item | Value |
|---|---|
| Host OS |  |
| Python |  |
| Torch |  |
| ONNX / ONNX Runtime |  |
| TensorFlow / TFLite |  |
| onnx2tf |  |
| RA / RUHMI / MERA tools |  |

## Model And Data

| Item | Value |
|---|---|
| Model | EthosSafeDet-A v1 |
| Input | batch=1, 320x320 / 256x256 |
| Classes | earbud, phial, bottle, phone, remote, tissue |
| Training manifest |  |
| Calibration manifest |  |
| Calibration image count |  |

## Artifacts

| Artifact | Path | SHA256 | Notes |
|---|---|---|---|
| checkpoint |  |  |  |
| ONNX reference |  |  |  |
| TFLite full-int8 |  |  |  |
| host MERA output |  |  |  |
| RUHMI dispatch log |  |  |  |
| static golden |  |  |  |

## Gate Results

| Gate | Command | PASS/FAIL | Evidence |
|---|---|---|---|
| ONNX ops | `check_onnx_ops --onnx <model.onnx>` |  |  |
| TFLite ops | `check_tflite_ops --tflite <model.tflite>` |  |  |
| PC ONNX vs TFLite | `compare_ethossafedet_reference ...` |  |  |
| host MERA | `run_host_mera_gate ...` |  |  |
| RUHMI dispatch | `inspect_ruhmi_dispatch --log <log>` |  |  |
| memory budget | `check_memory_budget --log <log>` |  |  |
| static golden | `make_static_golden ...` |  |  |

## Acceptance

| Item | Threshold | Result |
|---|---:|---|
| PC ONNX vs host MERA main class | identical |  |
| PC ONNX vs host MERA bbox IoU | >= 0.85 |  |
| board static vs host MERA main class | identical |  |
| board static vs host MERA bbox IoU | >= 0.85 |  |
| RUHMI `num_base_addr` | <= 8 |  |
| arena bytes | <= 2621440 |  |
| weights bytes | <= 1572864 |  |

## Result

PASS/FAIL:

## Next Step

1.
2.
