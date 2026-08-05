# CodeAgent：基于 FSD 训练单二维码 ordered_corners(8)

## 1. 目标与硬约束

在服务器现有 UltraFace/RFB FSD 工程中完成带方向的单二维码检测。

1. 基线只能是 `create_Mb_Tiny_RFB_fd_3_nodilation`。
2. 输入固定 `W×H=240×320`，Tensor 固定 `N×C×320×240`；横图顺时针旋转 90°。
3. 每张图恰好一个二维码。
4. 模型输出只能为 `confidence(2)+ordered_corners(8)`，不保留独立 `bbox(4)` 输出。
5. `P0/P1/P2/P3` 是二维码自身 TL/TR/BR/BL，顺时针；禁止按图像左上重新排序。
6. 不新增 landmark 分支、第二阶段、corner refinement 或额外 backbone。
7. 保留 backbone、RFB、extras、classification、四层 feature 和 prior；唯一必要变化是现有单一
   `regression_headers` 末层 `anchors×4 → anchors×8` 及 reshape `4 → 8`。
8. bbox 仅能从角点 min/max 派生，用于 prior matching、NMS 和指标，不是训练目标或模型输出。
9. 二维码代码放独立目录，不修改现有人脸检测训练、推理和量化文件。

本目录代码为 canonical implementation。不得把它改回 `bbox+landmark`，不得调用
`create_Mb_Tiny_RFB_fd_3_landmx_nodilation`。

## 2. 路径

```bash
FSD_ROOT=/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
QR_CODE_ROOT=$FSD_ROOT/qr_detection_ordered_corners
DATA_ROOT=/data/pub1/z00919662/dataset/qr_single_240x320
BACKGROUND_DIR=/data/pub1/z00919662/dataset/coco_ADE_12cls
OUTPUT_DIR=$FSD_ROOT/models/qr_fsd_240x320_corners8
```

实际 FSD 根目录必须包含 `vision/ssd/mb_tiny_RFB_fd_3.py`。如果路径不同，只修改环境变量。

## 3. 复制与只读预检

```bash
cd "$QR_CODE_ROOT"
python -V
python -c "import torch, cv2, numpy; print(torch.__version__, cv2.__version__, numpy.__version__)"
rg -n "def create_Mb_Tiny_RFB_fd_3_nodilation|def compute_header|view.*4" \
  "$FSD_ROOT/vision/ssd/mb_tiny_RFB_fd_3.py" \
  "$FSD_ROOT/vision/ssd/ssd.py"
python tests/test_qr_geometry.py
python tests/test_qr_model_adapter.py
```

确认原 factory 返回两个输出，原始回归为 `[N,4420,4]`。不要修改 FSD 源文件；`qr_model.py`
在构建二维码模型时对实例做最小适配。

## 4. 依赖

沿用现有 FSD conda 环境，不升级 torch、torchvision、opencv 或 numpy：

```bash
python -c "import qrcode" || python -m pip install qrcode==7.3.1 'Pillow>=8.0,<10.0'
```

## 5. 数据冒烟测试

```bash
SMOKE_ROOT=/data/pub1/z00919662/dataset/qr_single_240x320_smoke
python prepare_qr_dataset.py synthetic \
  --output "$SMOKE_ROOT" --background-dir "$BACKGROUND_DIR" \
  --rotate-landscape-cw --train-count 40 --val-count 8 --test-count 8
python validate_qr_dataset.py --data-root "$SMOKE_ROOT" --visualize 8
```

人工检查 `validation_preview`：P0 红、P1 黄、P2 蓝、P3 紫。旋转二维码中 P0 不一定是图像左上点。

### 真实数据标注

LabelMe 中每张图只画一个 label 为 `qrcode`/`qr`/`qr_code` 的四点 polygon。必须依次点击：

```text
P0 QR自身左上 → P1 QR自身右上 → P2 QR自身右下 → P3 QR自身左下
```

转换不会自动重排，因为自动选“图像左上点”会破坏方向：

```bash
python prepare_qr_dataset.py labelme \
  --input /absolute/path/to/labelme_real_qr \
  --output /data/pub1/z00919662/dataset/qr_real_240x320 \
  --rotate-cw landscape
python validate_qr_dataset.py \
  --data-root /data/pub1/z00919662/dataset/qr_real_240x320 --visualize 32
```

来自视频的数据必须按视频/片段分组划分，禁止相邻帧跨 train/val/test。

## 6. 正式合成数据

```bash
DATA_ROOT="$DATA_ROOT" BACKGROUND_DIR="$BACKGROUND_DIR" \
TRAIN_COUNT=40000 VAL_COUNT=4000 TEST_COUNT=4000 \
bash run_prepare_dataset.sh
```

合成数据用于初始化，最终必须用 2,000–5,000 张业务真实图微调，并用独立真实 test 汇报结果。

## 7. checkpoint 与模型自检

初始化 checkpoint 应来自当前单 Y 通道、`input_size=240` 的
`create_Mb_Tiny_RFB_fd_3_nodilation`。不使用 landmark、336 或 640×384 checkpoint。

`load_fd_pretrained` 的规则：

- backbone/RFB/extras/classification 必须 key 和 shape 完全匹配；
- 原 `regression_headers` 因 `4→8` 明确跳过并重新初始化；
- 任何其他 missing/unexpected/shape mismatch 立即停止。

## 8. 一轮训练冒烟测试

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" --data-root "$SMOKE_ROOT" \
  --checkpoint-dir "$OUTPUT_DIR/smoke" \
  --pretrained-fd /absolute/path/to/240_input_fsd_checkpoint.pth \
  --input-mode y --input-size-key 240 \
  --batch-size 4 --num-workers 0 --epochs 1 --gpus 0
```

必须看到：

```text
factory=create_Mb_Tiny_RFB_fd_3_nodilation
NCHW=(1,1,320,240)
priors=4420
confidence=(1,4420,2)
ordered_corners=(1,4420,8)
bbox_output=NONE
```

并确认 total/corner/classification loss 有限、反向传播和 checkpoint 保存成功。

## 9. 正式训练

```bash
export FSD_ROOT DATA_ROOT OUTPUT_DIR
export FD_CHECKPOINT=/absolute/path/to/240_input_fsd_checkpoint.pth
export CUDA_VISIBLE_DEVICES=0,1
export BATCH_SIZE=64 NUM_WORKERS=16 EPOCHS=200
mkdir -p "$OUTPUT_DIR"
cd "$QR_CODE_ROOT"
nohup bash run_train.sh > "$OUTPUT_DIR/nohup.out" 2>&1 &
```

先全网络微调。若灾难性遗忘，可用 `--freeze-base-net` 做对照，但不能改模型结构。

## 10. 测试与推理

```bash
python eval_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$OUTPUT_DIR/qr_fsd_best.pth" \
  --data-root "$DATA_ROOT" --split test --input-mode y --device cuda:0 \
  --output "$OUTPUT_DIR/test_metrics.json"

INPUT_PATH=/absolute/path/to/240x320/portrait_images \
CHECKPOINT="$OUTPUT_DIR/qr_fsd_best.pth" \
OUTPUT_PATH="$OUTPUT_DIR/infer_preview" bash run_infer.sh
```

若输入仍是 320×240 横图，直接运行 `infer_fsd_qr.py` 时添加 `--rotate-landscape-cw`。JSON 输出：

```text
score
ordered_corners: P0,P1,P2,P3
derived_bbox_xyxy: 仅后处理派生
```

评估角点误差时禁止 cyclic shift/minimum matching，否则会掩盖 90°/180°/270°方向错误。

至少汇报 detection rate、derived bbox IoU、polygon IoU、P0 error、ordered corner error、
success@5px/10px。合成 test 与真实业务 test 分开。

## 11. CodeAgent 最终报告

创建 `$OUTPUT_DIR/CODEAGENT_REPORT.md`，包含实际路径、环境/GPU、数据量、模型输出 shape、
checkpoint 加载检查、训练命令与 best epoch、合成/真实指标、至少 50 张可视化、失败样例归因，
以及后续 ONNX/C 侧输出由 4 改为 8、NMS bbox 从角点派生的同步修改。不要只写 `Task Complete`。
