# FSD 单二维码方向检测：仅 ordered_corners(8)

本实现以现有 `create_Mb_Tiny_RFB_fd_3_nodilation` 为唯一基线：

- 保留 backbone、RFB、extras、四层 feature map、classification head、prior 与单阶段流程；
- 不增加 landmark head、bbox head、第二阶段或额外 backbone；
- 将现有单一 `regression_headers` 从每个 anchor 的 `4` 个 bbox 回归量改为 `8` 个有序角点回归量；
- 模型最终只返回 `confidence(2) + ordered_corners(8)`；
- 匹配、NMS、可视化需要的水平 bbox 均从四角点取 min/max 临时计算，不是模型输出。

输入固定为 `W×H=240×320`，Tensor 为 `N×C×320×240`。横图先顺时针旋转 90°，点坐标同步执行：

```text
x' = old_height - 1 - y
y' = x
```

## 角点方向定义

```text
ordered_corners = [P0x,P0y,P1x,P1y,P2x,P2y,P3x,P3y]
P0 = 二维码自身的左上角
P1 = 二维码自身的右上角
P2 = 二维码自身的右下角
P3 = 二维码自身的左下角
```

`P0` 不是图像里最靠左上的点。二维码旋转 90° 后，P0 可能位于图像右上角；数据增强、旋转、
推理和评估均保留点的身份，绝不按图像位置重新编号。真实 LabelMe polygon 必须按 P0→P1→P2→P3
依次点击。

## 为什么无需 bbox(4)

GT 匹配时：

```python
derived_gt_bbox = [corners[:,0].min(), corners[:,1].min(),
                   corners[:,0].max(), corners[:,1].max()]
```

推理 NMS 同理从预测角点派生 bbox。因此 prior 仍负责候选位置和尺度，IoU/NMS 仍可复用，但网络不再
训练或输出单独的 bbox。

## 网络输出

竖图 feature map 为 `40×30、20×15、10×8、5×4`，共 `4420` 个 priors：

```text
confidence:      [N, 4420, 2]
ordered_corners: [N, 4420, 8]
```

`qr_model.py` 先调用原工程的 `create_Mb_Tiny_RFB_fd_3_nodilation`，再把每层回归 header 的末端
卷积从 `anchors×4` 替换为 `anchors×8`，并把原 `compute_header` 的 reshape 从 4 改成 8。
原 FSD checkpoint 加载时，除旧 `regression_headers` 外全部严格检查和加载；8 维回归头重新初始化。

## 数据格式

```text
qr_single_240x320/
  train/images/*.jpg
  train/annotations.jsonl
  val/images/*.jpg
  val/annotations.jsonl
  test/images/*.jpg
  test/annotations.jsonl
```

每行：

```json
{"image":"images/train_0000000.jpg","width":240,"height":320,"label":"qrcode","corners":[[209,20],[209,200],[59,200],[59,20]],"corner_order":["qr_top_left","qr_top_right","qr_bottom_right","qr_bottom_left"]}
```

标注不保存 bbox。代码支持单二维码合成数据，以及每图一个四点 polygon 的 LabelMe 真实数据转换。

## 快速运行

```bash
python3 -m pip install -r requirements_qr.txt
python3 tests/test_qr_geometry.py
python3 tests/test_qr_model_adapter.py

DATA_ROOT=/data/pub1/z00919662/dataset/qr_single_240x320 \
BACKGROUND_DIR=/data/pub1/z00919662/dataset/coco_ADE_12cls \
bash run_prepare_dataset.sh

FD_CHECKPOINT=/absolute/path/to/240_input_fsd.pth bash run_train.sh

INPUT_PATH=/absolute/path/to/240x320/images bash run_infer.sh
```

服务器完整步骤见 `CODEAGENT_RUN_QR_FSD.md`。
