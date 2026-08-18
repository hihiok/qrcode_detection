# CodeAgent：canonical 四数据集 FSD 多二维码 YUV 重训

## 1. 目标

使用已经统一为 `qr_ordered_corners_v1` 的 BarBeR、BoofCV、Mendeley 和 SYNTH：

1. 输入固定 W×H=240×320，Tensor 为 N×3×320×240 YUV444。
2. 每图支持0～N个二维码。
3. 输出 `confidence [N,4720,2] + ordered_corners [N,4720,8]`，没有 bbox head。
4. P0/P1/P2/P3 始终为二维码自身 TL/TR/BR/BL。
5. 从未受污染的旧单Y八角点 checkpoint 重新迁移训练。

旧的 `qr_fsd_multi_yuv` 曾把 BoofCV/Mendeley 的 `objects[]` 漏读为负样本，禁止作为
resume或最终模型。本次输出使用全新目录，不能覆盖旧结果。

## 2. 代理、SSL 和代码同步

执行任何 git/pip/curl 网络操作前，读取：

~~~bash
test -f /mnt/ssd1/z00919662/qrcode_detection/proxy.md
test -f /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md
sed -n '1,220p' /mnt/ssd1/z00919662/qrcode_detection/proxy.md
sed -n '1,220p' /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md
~~~

按文件设置 http_proxy/https_proxy 和 Git proxy。公司证书链导致 fetch 失败时，在 fetch 前执行：

~~~bash
git config --global http.sslVerify false
~~~

不得在日志或报告中回显代理账号或密码。

使用现有独立 worktree：

~~~bash
export PROJECT_ROOT=/mnt/ssd1/z00919662/qrcode_detection_multi_qr_yuv
cd "$PROJECT_ROOT"
git status --short
~~~

若输出非空，立即停止；不得 stash、clean、reset 或覆盖。工作区干净时：

~~~bash
git fetch origin agent/multi-qr-yuv
git checkout --detach origin/agent/multi-qr-yuv
git rev-parse HEAD
git status --short
~~~

## 3. canonical 数据前置条件

本次训练前必须已严格完成 `CODEAGENT_CANONICALIZE_QR_DATASETS.md`，并由人工确认预览。

~~~bash
export SOURCE_BASE=/mnt/ssd1/z00919662/qrcode_detection/dataset
export CANONICAL_ROOT=$SOURCE_BASE/qr_canonical_v1
export BARBER_ROOT=$CANONICAL_ROOT/barber
export BOOFCV_ROOT=$CANONICAL_ROOT/boofcv
export MENDELEY_ROOT=$CANONICAL_ROOT/mendeley
export SYNTH_ROOT=$CANONICAL_ROOT/synth
test -f "$CANONICAL_ROOT/conversion_report.json"
~~~

禁止把四个旧源目录直接传给训练脚本。每个 canonical 根目录必须包含三个 split、
`annotations.jsonl` 和图片软链接。所有行必须含：

~~~text
schema_version=qr_ordered_corners_v1
num_qrcodes=len(instances)
instances[].class_id=0
instances[].label=qrcode
instances[].corner_order=TL/TR/BR/BL
~~~

BoofCV和Mendeley任一split若仍为0 instances，立即停止。合成集必须为4000/400/400张。

## 4. 环境与测试

~~~bash
export PYTHON=/mnt/ssd1/z00919662/anaconda3/envs/ultraface/bin/python
export FSD_ROOT=/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
export OLD_QR_CHECKPOINT=$FSD_ROOT/models/qr_fsd_240x320_corners8/qr_fsd_best.pth
export OUTPUT_DIR=$FSD_ROOT/models/qr_fsd_multi_yuv_canonical_v1
cd "$PROJECT_ROOT"

test -f "$FSD_ROOT/vision/ssd/mb_tiny_RFB_fd_3.py"
test -f "$OLD_QR_CHECKPOINT"
if [ -e "$OUTPUT_DIR" ]; then
  echo "STOP: output already exists: $OUTPUT_DIR"
  exit 2
fi

"$PYTHON" -V
"$PYTHON" -c "import torch, cv2, numpy; print(torch.__version__, cv2.__version__, numpy.__version__)"
"$PYTHON" -m py_compile *.py tests/*.py
bash -n run_prepare_dataset.sh run_train.sh run_infer.sh
"$PYTHON" tests/test_qr_schema.py
"$PYTHON" tests/test_canonicalize_qr_datasets.py
"$PYTHON" tests/test_qr_geometry.py
"$PYTHON" tests/test_qr_model_adapter.py
"$PYTHON" tests/test_qr_dataset.py
"$PYTHON" tests/test_qr_loss.py
~~~

必须确认输入 `[1,3,320,240]`、4720 priors、confidence `[1,4720,2]`、
ordered corners `[1,4720,8]`、bbox output NONE。

## 5. 逐数据集验证

~~~bash
for dataset_root in "$BARBER_ROOT" "$BOOFCV_ROOT" "$MENDELEY_ROOT" "$SYNTH_ROOT"; do
  "$PYTHON" validate_qr_dataset.py --data-root "$dataset_root" --visualize 0
done
~~~

读取四份 `validation_report.json` 和总 `conversion_report.json`，报告各split图片数、实例数、
负样本数、实例直方图、JSON-TXT检查数、group_key检查和重复检查。检查源annotations哈希未变化。

## 6. canonical 冒烟数据与一轮训练

~~~bash
export SMOKE_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset_smoke_multi_canonical_v1
if [ -e "$SMOKE_ROOT" ]; then
  echo "STOP: smoke output already exists: $SMOKE_ROOT"
  exit 2
fi

"$PYTHON" prepare_qr_dataset.py synthetic \
  --output "$SMOKE_ROOT" \
  --background-dir /data/pub1/z00919662/dataset/coco_ADE_12cls \
  --train-count 40 --val-count 8 --test-count 8 \
  --max-qrs-per-image 5 --negative-ratio 0.15 \
  --rotate-landscape-cw
"$PYTHON" validate_qr_dataset.py --data-root "$SMOKE_ROOT" --visualize 16

mkdir -p "$OUTPUT_DIR/smoke"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" \
  --data-root "$SMOKE_ROOT" \
  --checkpoint-dir "$OUTPUT_DIR/smoke" \
  --resume "$OLD_QR_CHECKPOINT" \
  --input-mode yuv --input-size-key 240 \
  --batch-size 4 --num-workers 0 --epochs 1 --gpus 0
~~~

确认多GT匹配、显式负样本classification loss和反向传播正常。失败时不得开始正式训练。

## 7. 正式训练

~~~bash
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0,1 nohup "$PYTHON" -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" \
  --data-root "$BARBER_ROOT" \
  --data-root "$BOOFCV_ROOT" \
  --data-root "$MENDELEY_ROOT" \
  --data-root "$SYNTH_ROOT" \
  --checkpoint-dir "$OUTPUT_DIR" \
  --resume "$OLD_QR_CHECKPOINT" \
  --input-mode yuv --input-size-key 240 \
  --batch-size 64 --num-workers 16 --epochs 200 --gpus 0,1 \
  > "$OUTPUT_DIR/nohup.out" 2>&1 &
~~~

首层迁移必须明确打印：旧Y权重复制到Y，U/V置零；除首层1→3外严格加载。
使用验证loss最低的 `qr_fsd_best.pth`，不能只使用最后epoch。

## 8. 分数据集评估

~~~bash
for dataset_root in "$BARBER_ROOT" "$BOOFCV_ROOT" "$MENDELEY_ROOT" "$SYNTH_ROOT"; do
  dataset_name=$(basename "$dataset_root")
  "$PYTHON" eval_fsd_qr.py \
    --fsd-repo "$FSD_ROOT" \
    --checkpoint "$OUTPUT_DIR/qr_fsd_best.pth" \
    --data-root "$dataset_root" --split test \
    --input-mode yuv --device cuda:0 \
    --score-threshold 0.5 --match-iou-threshold 0.5 \
    --max-detections 20 \
    --output "$OUTPUT_DIR/test_metrics_${dataset_name}.json"
done
~~~

四个数据集分别报告TP/FP/FN、precision、recall、F1、负样本误检率、bbox/polygon IoU、
P0与ordered-corner误差、success@5px/10px。预测和GT必须一对一匹配。

## 9. MP4推理

先只读查找原始视频：

~~~bash
find /mnt/ssd1/z00919662/qrcode_detection -type f -iname '*vrtest*.mp4' -print
~~~

不得把已经画框的 `vrtest_output.mp4` 当输入。若找不到原始MP4，跳过并明确要求人工提供路径。

找到原始视频后：

~~~bash
"$PYTHON" infer_video.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$OUTPUT_DIR/qr_fsd_best.pth" \
  --input /actual/raw/vrtest.mp4 \
  --output /mnt/ssd1/z00919662/qrcode_detection/video_output/vrtest_multi_yuv_canonical_v1.mp4 \
  --score-threshold 0.8 --nms-threshold 0.3 --max-detections 20
~~~

## 10. 最终报告

在 `$OUTPUT_DIR/CODEAGENT_REPORT.md` 报告实际commit和路径、四数据集完整统计、schema校验、
所有测试、YUV和checkpoint迁移、best epoch、四组测试指标、MP4路径及台球/球网等
hard-negative结果。列出所有需要人工处理的事项，不得修改或覆盖四个源数据集。
