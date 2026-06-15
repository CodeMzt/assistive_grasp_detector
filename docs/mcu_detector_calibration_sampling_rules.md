# EthosSafeDet Calibration Sampling Rules

EthosSafeDetV2 full-int8 calibration must use a fixed manifest of real OV5640 VGA 640x480 board-camera images. Calibration image selection proves input provenance only; it does not prove class/bbox/orientation accuracy.

## Hard Rules

- Use 200-500 images.
- Use only images listed in the accepted EthosSafeDet manifest for the candidate run.
- Do not use random npy arrays, generated noise, screenshots, or a single smoke-test image.
- Preserve the selected image list, seed, source manifest path, and generation time.

## Command

```powershell
build_ethossafedet_calibration `
  --manifest <ethossafedet_manifest.jsonl> `
  --out <calibration.json> `
  --target-count 320 `
  --seed 0
```

The output schema is `ethossafedet_calibration_v1`; each item stores the absolute image path, split, negative flag, and class ids present in that frame. Orientation coverage is audited in the training/golden manifests, not inferred from calibration selection alone.

## Coverage Guidance

The first gate only enforces count and real-image provenance. When choosing or auditing a calibration set, record coverage of:

- exposure and brightness,
- target scale,
- all 7 classes,
- negative/background frames,
- clutter or strong texture backgrounds.

Do not claim any board precision, RUHMI dispatch, arena, or latency result from calibration selection alone; those remain experiment facts only after the gate logs exist.
