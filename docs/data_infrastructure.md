# Data Infrastructure v0

本仓库的数据基础设施只消费 `AssistiveGraspAnnotator` 的磁盘数据契约，不复制 GUI 标注器源码，也不把标注器作为子模块。

## 自采数据契约

推荐的数据根目录形态：

```text
dataset_root/
  classes.yaml
  images/
    board_vga/
      000001.jpg
  annotations/
    board_vga/
      000001.json
  splits/
  generated/
    detector_yolo/
    target_maps/
```

关键约定：

- `images/board_vga/*.jpg` 是 OV5640 VGA 640x480 RGB 文件形式的训练侧母图。
- `annotations/board_vga/*.json` 与 `images/board_vga/*.jpg` 按相同相对路径对应。
- JSON 里的 `bbox_xyxy` 和 grasp `points` 都必须是 VGA 640x480 坐标。
- `classes.yaml` 使用 `configs/classes/assistive_grasp_v0.yaml` 的结构；当前类别表是 v0/未冻结。
- 标注器导出的 Model B `generated/target_maps/**/*.npz` 是训练 target 的来源，主仓库不复刻 target map 生成算法。

## CLI

```powershell
validate_self_dataset --dataset D:\path\to\dataset
build_model_a_yolo --dataset D:\path\to\dataset --out data\generated\model_a\self_v0
index_model_b_targets --target-maps D:\path\to\dataset\generated\target_maps --out data\generated\manifests\model_b_self_v0.jsonl
prepare_coco_subset --coco-root data\external\coco2017 --config configs\coco\model_a_coco_subset_v0.yaml --out data\generated\model_a\coco_subset_v0
```

`validate_self_dataset` 会把非 640x480 图像标为 warning。warning 不代表板端事实已验证；所有板端延迟、内存和后处理成本仍以实验记录为准。

## COCO 用法边界

COCO 只用于 Model A 的检测预热或泛化实验：

- `cell phone -> phone_other`
- `cup -> cup_other`
- `bottle -> bottle_other`
- `book -> book`
- `remote` 默认排除，因为 v0 类别表没有 `remote_other`

COCO 不用于 Model B，不生成 grasp rectangle，也不映射到 `phone_A`、`cup_A`、`bottle_A` 等身份类。
