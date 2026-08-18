# CodeAgent 指令二：不改网络结构的QR优化、重训与一次性严格评估

## 1. 唯一目标和禁止事项

从GitHub下载我已经写好的代码，在冻结的V2 manifest上完成：

1. 坐标/YUV/anchor/标签审计；
2. 数据源平衡采样；
3. 不改网络结构的角点loss优化；
4. 训练和公共validation阈值选择；
5. 公共test的raw与OpenCV ROI精修评估；
6. checkpoint、阈值和后处理全部冻结后，才对`fore.mp4`和`vrtest.mp4`各运行一次最终推理。

禁止CodeAgent自行编写或修改代码。若代码、数据或环境不符合要求，立即停止并报告；不得patch、
commit、push、降低门限或绕过检查。

模型结构必须保持：`Mb_Tiny_RFB_fd_3 nodilation`、输入`3×320×240 YUV444`、4720 priors、
输出`confidence[4720,2] + ordered_corners[4720,8]`、没有bbox head。

## 2. 代理与SSL（必须先做）

代理凭据只从服务器本地读取，不得写进GitHub、日志或最终报告：

```bash
export SOURCE_PROJECT=/mnt/ssd1/z00919662/qrcode_detection
test -f "$SOURCE_PROJECT/proxy.md"
test -f "$SOURCE_PROJECT/CODEAGENT_DISABLE_SSL.md"
sed -n '1,220p' "$SOURCE_PROJECT/proxy.md"
sed -n '1,220p' "$SOURCE_PROJECT/CODEAGENT_DISABLE_SSL.md"

export http_proxy="<从proxy.md读取>"
export https_proxy="<从proxy.md读取>"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org"
```

## 3. 下载已准备代码

```bash
export CODE_ROOT=/mnt/ssd1/z00919662/qrcode_detection_strict_v2_opt_code
export REPO_URL=https://github.com/hihiok/qrcode_detection.git
export BRANCH=agent/qr-strict-v2-optimization

if [ -e "$CODE_ROOT" ]; then
  echo "STOP: $CODE_ROOT already exists; do not overwrite"
  exit 2
fi
git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$CODE_ROOT"
cd "$CODE_ROOT"
git status --short
git rev-parse HEAD
```

工作区必须干净。不要在仓库里生成checkpoint或数据。

## 4. 固定输入、输出和严格评估清单

```bash
export PYTHON=/mnt/ssd1/z00919662/anaconda3/envs/ultraface/bin/python
export FSD_ROOT=/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
export V2_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_strict_v2
export DATASET_MANIFEST=$V2_ROOT/QR_DATASET_V2_MANIFEST.json
export STRICT_MANIFEST=$V2_ROOT/STRICT_EVAL_MANIFEST.json
export OLD_BEST=$FSD_ROOT/models/qr_fsd_multi_yuv_canonical_v1/qr_fsd_best.pth
export OUTPUT_DIR=$FSD_ROOT/models/qr_fsd_multi_yuv_strict_v2

test -f "$DATASET_MANIFEST"
test -f "$STRICT_MANIFEST"
test -f "$OLD_BEST"
if [ -e "$OUTPUT_DIR" ]; then
  echo "STOP: output already exists: $OUTPUT_DIR"
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
```

数据集指令一若仍为`WAITING_FOR_HUMAN_HARD_NEGATIVE_REVIEW`，这里必须停止。

## 5. 代码、manifest和结构审计

```bash
cd "$CODE_ROOT"
"$PYTHON" -m py_compile *.py tests/*.py
for test_file in tests/test_*.py; do "$PYTHON" "$test_file" || exit 1; done
"$PYTHON" dataset_v2_manifest.py verify --manifest "$DATASET_MANIFEST"

"$PYTHON" audit_qr_training_pipeline.py \
  --manifest "$DATASET_MANIFEST" --split train \
  --max-images-per-source 5000 --preview-per-source 16 \
  --output-dir "$OUTPUT_DIR/pretrain_audit"
```

读取`pipeline_audit.json`并确认：

- 480×408、1280×720等坐标round-trip最大误差小于`1e-3 px`；
- BGR→YUV使用`cv2.COLOR_BGR2YUV`、YUV444、`float32/255`；
- 4层feature maps为40×30、20×15、10×8、5×4；
- priors为4720；
- 所有实例有正anchor，所有角点在240×320内；
- strict-eval leakage为0。

## 6. 一轮独立smoke训练

smoke只能使用程序生成的新临时数据，不能使用两个业务视频：

```bash
export SMOKE_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset_smoke_strict_v2
if [ -e "$SMOKE_ROOT" ]; then
  echo "STOP: smoke dataset already exists: $SMOKE_ROOT"
  exit 2
fi

"$PYTHON" prepare_qr_dataset.py synthetic \
  --output "$SMOKE_ROOT" --train-count 40 --val-count 8 --test-count 8 \
  --max-qrs-per-image 3 --single-qr-probability 0.90 \
  --negative-ratio 0.25 --decoy-probability 0.70 --seed 20260820

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" --data-root "$SMOKE_ROOT" \
  --checkpoint-dir "$OUTPUT_DIR/smoke" --resume "$OLD_BEST" \
  --batch-size 4 --num-workers 0 --epochs 1 --gpus 0 \
  --lr 1e-3 --corner-weight 2.0 --normalized-corner-weight 4.0 \
  --edge-weight 1.0 --orientation-weight 0.25 \
  --neg-pos-ratio 3 --min-negatives-per-image 64
```

确认classification、encoded corner、normalized corner、edge、orientation loss均有限，负样本batch有
classification梯度，输出形状完全未变。

## 7. 正式训练

训练从当前YUV ordered-corner best checkpoint继续，不从业务视频调参。V2采样权重由manifest冻结；
validation checkpoint selection使用各source的macro平均，不能由2万张synthetic支配。

```bash
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES=0,1 nohup "$PYTHON" -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --samples-per-epoch 40000 \
  --checkpoint-dir "$OUTPUT_DIR" --resume "$OLD_BEST" \
  --input-mode yuv --input-size-key 240 \
  --batch-size 64 --num-workers 16 --epochs 200 --gpus 0,1 \
  --lr 1e-3 --momentum 0.9 --weight-decay 5e-4 \
  --milestones 80,130,170 --gamma 0.1 \
  --iou-threshold 0.35 \
  --corner-weight 2.0 --normalized-corner-weight 4.0 \
  --edge-weight 1.0 --orientation-weight 0.25 \
  --classification-weight 1.0 --neg-pos-ratio 3 \
  --min-negatives-per-image 64 --gradient-clip 10 \
  > "$OUTPUT_DIR/nohup.out" 2>&1 &
```

记录PID和GPU。开始后检查前200个step，确认不是OOM、NaN或全零loss。训练结束使用
`qr_fsd_best.pth`，其选择依据必须是`macro_source_val_total`，不能使用latest代替。

200 epoch结束后如果best仍在最后5个epoch，不得擅自继续训练；在报告中说明可能未完全收敛并等待用户决定。

## 8. 只用公共validation冻结阈值和最终后处理

此步骤之前绝对不能运行两个业务视频：

```bash
export BEST=$OUTPUT_DIR/qr_fsd_best.pth
export FROZEN_CONFIG=$OUTPUT_DIR/FROZEN_INFERENCE_CONFIG.json
test -f "$BEST"

"$PYTHON" select_qr_threshold.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$BEST" \
  --dataset-manifest "$DATASET_MANIFEST" --device cuda:0 \
  --thresholds 0.30,0.40,0.50,0.60,0.70,0.80,0.90 \
  --nms-threshold 0.30 --match-iou-threshold 0.50 \
  --max-detections-validation 20 --max-detections-final 1 \
  --opencv-refine-final --output "$FROZEN_CONFIG"

test -s "$FROZEN_CONFIG"
export SCORE_THRESHOLD=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["score_threshold"])' \
  "$FROZEN_CONFIG")
```

`FROZEN_INFERENCE_CONFIG.json`必须写明：selection只用了manifest的val splits、
`strict_eval_videos_used=false`、checkpoint SHA-256、score/NMS阈值、max detections和OpenCV精修参数。

## 9. 公共test评估raw网络和hybrid精修

```bash
mkdir -p "$OUTPUT_DIR/public_test_raw" "$OUTPUT_DIR/public_test_refined"

for dataset_root in \
  /mnt/ssd1/z00919662/qrcode_detection/dataset/qr_canonical_v1/barber \
  /mnt/ssd1/z00919662/qrcode_detection/dataset/qr_canonical_v1/boofcv \
  /mnt/ssd1/z00919662/qrcode_detection/dataset/qr_canonical_v1/mendeley \
  "$V2_ROOT/synth_v2" \
  "$V2_ROOT/negative_v2"; do
  dataset_name=$(basename "$dataset_root")
  "$PYTHON" eval_fsd_qr.py \
    --fsd-repo "$FSD_ROOT" --checkpoint "$BEST" \
    --data-root "$dataset_root" --split test --device cuda:0 \
    --score-threshold "$SCORE_THRESHOLD" --match-iou-threshold 0.50 \
    --nms-threshold 0.30 --max-detections 20 \
    --output "$OUTPUT_DIR/public_test_raw/${dataset_name}.json"

  "$PYTHON" eval_fsd_qr.py \
    --fsd-repo "$FSD_ROOT" --checkpoint "$BEST" \
    --data-root "$dataset_root" --split test --device cuda:0 \
    --score-threshold "$SCORE_THRESHOLD" --match-iou-threshold 0.50 \
    --nms-threshold 0.30 --max-detections 20 --opencv-refine \
    --refine-roi-expand 0.18 --refine-min-iou 0.20 --refine-max-shift 0.40 \
    --output "$OUTPUT_DIR/public_test_refined/${dataset_name}.json"
done
```

分别报告P/R/F1、negative image FPR、bbox/polygon IoU、P0误差、mean ordered-corner error、
success@2/4/5/8/10 px。不得使用public test重新选择阈值或决定是否启用精修；最终配置已在上一步冻结。

## 10. 最后才运行两个锁定业务视频

这是一次性最终回归。运行后不能根据结果修改checkpoint、阈值、padding、NMS、精修参数或重新选择best。
技术崩溃可用完全相同配置重跑，但必须记录原因。

```bash
export FORE_VIDEO=/mnt/ssd1/z00919662/qrcode_detection/dataset/from_chenshuo/fore.mp4
export VRTEST_VIDEO=/mnt/ssd1/z00919662/qrcode_detection/dataset/from_chenshuo/vrtest.mp4
export VIDEO_OUT=/mnt/ssd1/z00919662/qrcode_detection/video_output/strict_v2_final
mkdir -p "$VIDEO_OUT"

for video in "$FORE_VIDEO" "$VRTEST_VIDEO"; do
  "$PYTHON" strict_eval_guard.py verify-final-input \
    --manifest "$STRICT_MANIFEST" --video "$video"
done

"$PYTHON" infer_video.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$BEST" \
  --input "$FORE_VIDEO" --output "$VIDEO_OUT/fore_strict_v2_refined.mp4" \
  --device cuda:0 --score-threshold "$SCORE_THRESHOLD" \
  --nms-threshold 0.30 --max-detections 1 \
  --pad-to-portrait-3x4 --pad-value 127 --opencv-refine \
  --refine-roi-expand 0.18 --refine-min-iou 0.20 --refine-max-shift 0.40 \
  --strict-eval-manifest "$STRICT_MANIFEST"

"$PYTHON" infer_video.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$BEST" \
  --input "$VRTEST_VIDEO" --output "$VIDEO_OUT/vrtest_strict_v2_refined.mp4" \
  --device cuda:0 --score-threshold "$SCORE_THRESHOLD" \
  --nms-threshold 0.30 --max-detections 1 \
  --pad-to-portrait-3x4 --pad-value 127 --opencv-refine \
  --refine-roi-expand 0.18 --refine-min-iou 0.20 --refine-max-shift 0.40 \
  --strict-eval-manifest "$STRICT_MANIFEST"
```

只做结果记录，不再调参。视频没有逐帧GT，因此detections/frame不能冒充precision或recall。

## 11. 最终报告

生成`$OUTPUT_DIR/CODEAGENT_STRICT_V2_REPORT.md`，包括：

- commit、环境、GPU、数据manifest和checkpoint SHA-256；
- 输出结构/参数量与旧模型一致的证据；
- pipeline audit结果；
- 训练曲线、best epoch、各source val loss；
- frozen threshold的公共validation选择依据；
- 每个公共test数据集raw/refined完整指标；
- 两个业务视频的一次性输出、帧数、速度、检测帧比例、总检测数；
- 明确说明业务视频无GT，不能报告真实P/R/F1；
- 明确说明看完业务结果后没有进行任何调参或重跑；
- 所有失败、异常以及需要用户人工处理的事项。

完成标志：`QR_STRICT_V2_FINAL_EVAL_COMPLETE`。
