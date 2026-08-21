# CodeAgent：训练并评估两阶段QR网络

## 1. 目标和固定结构

从GitHub下载已经准备好的代码，训练两个完全相同结构的FSD网络：

- Stage 1：全图输入，检测二维码几何四边形；目标顺序为图片几何`TL,TR,BR,BL`，不学习二维码自身方向；
- Stage 2：单二维码ROI输入，输出语义`P0,P1,P2,P3`，并拒绝负ROI；
- 两个模型均为相同的`Mb_Tiny_RFB_fd_3 nodilation`及`confidence + corners(8)` head，没有bbox head；
- Stage 1输入`3×320×240 YUV444`，4720 priors；Stage 2输入`3×112×112 YUV444`，
  feature maps为14×14、7×7、4×4、2×2，共779 priors；
- 输入空间尺寸不同只改变feature map位置数，不改变卷积层、head参数结构或state-dict tensor shape；
- 最终将Stage 2坐标用逆单应矩阵映射回原图，画出P0～P3。

禁止CodeAgent自行写或修改代码、commit、push、降低门限或绕过检查。若需要代码变更，停止并把日志、
diff和复现命令同步给用户，由用户交给代码作者处理。

`fore.mp4`和`vrtest.mp4`只允许在checkpoint、阈值、crop margin和后处理全部冻结后各运行一次。

## 2. 代理和SSL（clone前必须执行）

代理凭据只从服务器本地读取，不得写进GitHub或报告：

```bash
export SOURCE_PROJECT=/mnt/ssd1/z00919662/qrcode_detection
test -s "$SOURCE_PROJECT/proxy.md"
test -s "$SOURCE_PROJECT/CODEAGENT_DISABLE_SSL.md"
export QR_PROXY_URL=$(python - <<'PY'
import re
s=open('/mnt/ssd1/z00919662/qrcode_detection/proxy.md').read()
m=re.search(r'https?://[^\s"`]+', s)
if not m: raise SystemExit('STOP: proxy URL not found')
print(m.group(0))
PY
)
export http_proxy="$QR_PROXY_URL"
export https_proxy="$QR_PROXY_URL"
export HTTP_PROXY="$QR_PROXY_URL"
export HTTPS_PROXY="$QR_PROXY_URL"
git config --global http.proxy "$QR_PROXY_URL"
git config --global https.proxy "$QR_PROXY_URL"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=true
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org"
```

## 3. 下载代码和固定路径

```bash
export CODE_ROOT=/mnt/ssd1/z00919662/qrcode_detection_two_stage_train_code
export REPO_URL=https://github.com/hihiok/qrcode_detection.git
export BRANCH=agent/qr-two-stage-ordered-corners-v1
if [ -e "$CODE_ROOT" ]; then echo "STOP: $CODE_ROOT already exists"; exit 2; fi
git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$CODE_ROOT"
cd "$CODE_ROOT"
git status --short
git rev-parse HEAD

export PYTHON=/mnt/ssd1/z00919662/anaconda3/envs/ultraface/bin/python
export FSD_ROOT=/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
export SOURCE_DATA=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_train_combined_v3/unified
export STAGE2_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_stage2_ordered_v1
export STAGE2_MANIFEST=$STAGE2_ROOT/QR_STAGE2_DATASET_MANIFEST.json
export STAGE2_APPROVAL=$STAGE2_ROOT/APPROVED_STAGE2_DATASET.txt
export STRICT_MANIFEST=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_strict_v2/STRICT_EVAL_MANIFEST.json
export OLD_BEST=$FSD_ROOT/models/qr_fsd_multi_yuv_canonical_v1/qr_fsd_best.pth
export OUTPUT_ROOT=$FSD_ROOT/models/qr_fsd_two_stage_ordered_v1
export STAGE1_OUT=$OUTPUT_ROOT/stage1_geometry
export STAGE2_OUT=$OUTPUT_ROOT/stage2_semantic

test -x "$PYTHON"
test -s "$STAGE2_MANIFEST"
test -s "$STAGE2_APPROVAL"
test -s "$STRICT_MANIFEST"
test -s "$OLD_BEST"
if [ -e "$OUTPUT_ROOT" ]; then echo "STOP: output already exists: $OUTPUT_ROOT"; exit 2; fi
mkdir -p "$STAGE1_OUT" "$STAGE2_OUT"
```

缺少`APPROVED_STAGE2_DATASET.txt`时必须停止，不能由CodeAgent代替人工批准。

## 4. 全部预检

```bash
cd "$CODE_ROOT"
"$PYTHON" -m py_compile *.py tests/*.py
for test_file in tests/test_*.py; do "$PYTHON" "$test_file" || exit 1; done
"$PYTHON" prepare_qr_stage2_dataset.py verify --manifest "$STAGE2_MANIFEST"
"$PYTHON" strict_eval_guard.py check \
  --manifest "$STRICT_MANIFEST" \
  --dataset-root "$SOURCE_DATA" --dataset-root "$STAGE2_ROOT"
```

必须看到geometry round-trip、stage2 build/verify、模型adapter、loss和strict guard测试全部PASS。

## 5. 训练Stage 1：只学几何位置

Stage 1直接使用现有数据。loader只为Stage 1把四点重排为图片几何`TL,TR,BR,BL`；训练和推理都不把
Stage 1点序号解释成二维码自身方向。

```bash
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES=0,1 nohup "$PYTHON" -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" --data-root "$SOURCE_DATA" \
  --checkpoint-dir "$STAGE1_OUT" --resume "$OLD_BEST" \
  --target-mode geometry --input-mode yuv --input-size-key 240 \
  --batch-size 64 --num-workers 16 --epochs 120 --gpus 0,1 \
  --lr 1e-3 --momentum 0.9 --weight-decay 5e-4 \
  --milestones 50,85,105 --gamma 0.1 --iou-threshold 0.35 \
  --corner-weight 2.0 --normalized-corner-weight 4.0 \
  --edge-weight 1.0 --orientation-weight 0.25 \
  --classification-weight 1.0 --neg-pos-ratio 3 \
  --min-negatives-per-image 64 --gradient-clip 10 \
  > "$STAGE1_OUT/nohup.out" 2>&1 &
```

记录PID/GPU。前200 step检查OOM、NaN、数据路径和loss；结束后只能使用
`$STAGE1_OUT/qr_fsd_best.pth`。

## 6. 训练Stage 2：ROI内语义方向和精角点

Stage 2从旧semantic checkpoint初始化，而不是从Stage 1 geometry checkpoint初始化。

```bash
cd "$CODE_ROOT"
CUDA_VISIBLE_DEVICES=0,1 nohup "$PYTHON" -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" --data-root "$STAGE2_ROOT" \
  --checkpoint-dir "$STAGE2_OUT" --resume "$OLD_BEST" \
  --dataset-kind stage2_roi --target-mode semantic \
  --input-mode yuv --input-size-key 112 --input-width 112 --input-height 112 \
  --batch-size 64 --num-workers 16 --epochs 100 --gpus 0,1 \
  --lr 1e-3 --momentum 0.9 --weight-decay 5e-4 \
  --milestones 40,70,90 --gamma 0.1 --iou-threshold 0.35 \
  --corner-weight 2.0 --normalized-corner-weight 5.0 \
  --edge-weight 1.0 --orientation-weight 0.50 \
  --classification-weight 1.0 --neg-pos-ratio 3 \
  --min-negatives-per-image 64 --gradient-clip 10 \
  > "$STAGE2_OUT/nohup.out" 2>&1 &
```

确认日志显示输入`N×3×112×112`、779 priors；训练只读取预生成ROI，不执行原图裁剪或在线几何变形。
确认负ROI产生classification梯度，正ROI的P0误差持续下降。结束后只能使用
`$STAGE2_OUT/qr_fsd_best.pth`。若任一best出现在最后5个epoch，不得自行延长训练。

## 7. 验证结构完全一致并做smoke

```bash
export STAGE1_BEST=$STAGE1_OUT/qr_fsd_best.pth
export STAGE2_BEST=$STAGE2_OUT/qr_fsd_best.pth
test -s "$STAGE1_BEST"; test -s "$STAGE2_BEST"
mkdir -p "$OUTPUT_ROOT/smoke_images" "$OUTPUT_ROOT/smoke_output"
find "$SOURCE_DATA/val" -type f \( -iname '*.jpg' -o -iname '*.png' \) | head -n 5 | \
  xargs -I{} cp -n {} "$OUTPUT_ROOT/smoke_images/"
"$PYTHON" infer_qr_two_stage.py \
  --fsd-repo "$FSD_ROOT" \
  --stage1-checkpoint "$STAGE1_BEST" --stage2-checkpoint "$STAGE2_BEST" \
  --input "$OUTPUT_ROOT/smoke_images" --output "$OUTPUT_ROOT/smoke_output" \
  --device cuda:0 --stage1-score-threshold 0.30 \
  --stage2-score-threshold 0.40 --max-detections 20
```

日志必须出现`TWO_STAGE_ARCHITECTURE_MATCH`。检查输出图P0～P3和JSON逆映射坐标。smoke只确认可运行，
不能据此选择最终阈值。

## 8. 仅用validation冻结两阶段阈值

```bash
export FROZEN_CONFIG=$OUTPUT_ROOT/FROZEN_TWO_STAGE_CONFIG.json
"$PYTHON" select_qr_two_stage_thresholds.py \
  --fsd-repo "$FSD_ROOT" \
  --stage1-checkpoint "$STAGE1_BEST" --stage2-checkpoint "$STAGE2_BEST" \
  --validation-root "$SOURCE_DATA" --device cuda:0 \
  --stage1-thresholds 0.30,0.40,0.50,0.60,0.70 \
  --stage2-thresholds 0.40,0.50,0.60,0.70,0.80 \
  --nms-threshold 0.30 --match-iou-threshold 0.50 \
  --crop-margin 0.20 --stage2-min-geometry-iou 0.20 \
  --max-detections-validation 20 --max-detections-final 1 \
  --output "$FROZEN_CONFIG"
test -s "$FROZEN_CONFIG"
export S1_THR=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["stage1_score_threshold"])' "$FROZEN_CONFIG")
export S2_THR=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["stage2_score_threshold"])' "$FROZEN_CONFIG")
```

禁止用test或业务视频选择阈值、crop margin或是否启用OpenCV refine。首轮基线不启用OpenCV refine，
用于确认两阶段网络本身的收益。

## 9. 公共test严格评估

```bash
mkdir -p "$OUTPUT_ROOT/public_test"
"$PYTHON" eval_qr_two_stage.py \
  --fsd-repo "$FSD_ROOT" \
  --stage1-checkpoint "$STAGE1_BEST" --stage2-checkpoint "$STAGE2_BEST" \
  --data-root "$SOURCE_DATA" --split test --device cuda:0 \
  --stage1-score-threshold "$S1_THR" --stage2-score-threshold "$S2_THR" \
  --nms-threshold 0.30 --match-iou-threshold 0.50 \
  --crop-margin 0.20 --stage2-min-geometry-iou 0.20 --max-detections 20 \
  --output "$OUTPUT_ROOT/public_test/two_stage.json"
```

报告P/R/F1、negative FPR、bbox/polygon IoU、P0误差、mean ordered-corner error及
success@2/4/5/8/10px。test只报告，不能反向调参。

## 10. 最后才运行两个锁定业务视频

```bash
export FORE=/mnt/ssd1/z00919662/qrcode_detection/dataset/from_chenshuo/fore.mp4
export VRTEST=/mnt/ssd1/z00919662/qrcode_detection/dataset/from_chenshuo/vrtest.mp4
export VIDEO_OUT=/mnt/ssd1/z00919662/qrcode_detection/video_output/two_stage_ordered_v1_final
mkdir -p "$VIDEO_OUT"
for video in "$FORE" "$VRTEST"; do
  "$PYTHON" strict_eval_guard.py verify-final-input --manifest "$STRICT_MANIFEST" --video "$video"
done

"$PYTHON" infer_qr_two_stage_video.py \
  --fsd-repo "$FSD_ROOT" --stage1-checkpoint "$STAGE1_BEST" --stage2-checkpoint "$STAGE2_BEST" \
  --input "$FORE" --output "$VIDEO_OUT/fore_two_stage.mp4" --device cuda:0 \
  --stage1-score-threshold "$S1_THR" --stage2-score-threshold "$S2_THR" \
  --nms-threshold 0.30 --max-detections 1 --crop-margin 0.20 \
  --stage2-min-geometry-iou 0.20 --pad-to-portrait-3x4 --pad-value 127 \
  --strict-eval-manifest "$STRICT_MANIFEST"

"$PYTHON" infer_qr_two_stage_video.py \
  --fsd-repo "$FSD_ROOT" --stage1-checkpoint "$STAGE1_BEST" --stage2-checkpoint "$STAGE2_BEST" \
  --input "$VRTEST" --output "$VIDEO_OUT/vrtest_two_stage.mp4" --device cuda:0 \
  --stage1-score-threshold "$S1_THR" --stage2-score-threshold "$S2_THR" \
  --nms-threshold 0.30 --max-detections 1 --crop-margin 0.20 \
  --stage2-min-geometry-iou 0.20 --pad-to-portrait-3x4 --pad-value 127 \
  --strict-eval-manifest "$STRICT_MANIFEST"
```

业务视频没有逐帧GT，检测帧比例不能冒充precision/recall。看完后禁止修改参数重跑；下一版必须使用
新版本名并保留本次结果。

## 11. 最终报告

生成`$OUTPUT_ROOT/CODEAGENT_QR_TWO_STAGE_REPORT.md`，包括commit、环境、GPU、两阶段参数量和
state-dict shape一致证据、两阶段训练曲线和best epoch、数据manifest与人工批准文件SHA-256、
validation阈值网格、public test指标、业务视频帧数/速度/Stage 1检测数/Stage 2接受与拒绝数、所有异常，
并声明两个业务视频未参与训练、验证和阈值选择。

完成标志：`QR_TWO_STAGE_FINAL_EVAL_COMPLETE`。
