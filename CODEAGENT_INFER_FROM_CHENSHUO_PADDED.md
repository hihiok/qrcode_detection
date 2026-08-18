# CodeAgent：from_chenshuo 两个 MP4 居中 Pad 3:4 后二维码推理

## 1. 目标和禁止事项

对 `/mnt/ssd1/z00919662/qrcode_detection/dataset/from_chenshuo` 顶层的两个原始 MP4
逐帧居中补边到精确的 3:4 竖版画布，再用 canonical 四数据集训练得到的最佳 YUV 模型推理。
原始像素不得拉伸、裁剪或旋转。

必须同时输出未画框的 3:4 padded 视频和画框结果。禁止覆盖输入视频、训练 checkpoint、
数据集或已有输出；禁止自行修改代码。

## 2. 代理、SSL 和代码同步

执行网络操作前读取服务器配置：

~~~bash
test -f /mnt/ssd1/z00919662/qrcode_detection/proxy.md
test -f /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md
sed -n '1,220p' /mnt/ssd1/z00919662/qrcode_detection/proxy.md
sed -n '1,220p' /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md

export http_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
git config --global https.proxy "http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
git config --global http.sslVerify false
~~~

报告中不得回显代理账号或密码。

~~~bash
export PROJECT_ROOT=/mnt/ssd1/z00919662/qrcode_detection_multi_qr_yuv
cd "$PROJECT_ROOT"
git status --short
~~~

若输出非空立即停止；不得 stash、clean、reset 或覆盖。工作区干净时：

~~~bash
git fetch origin agent/multi-qr-yuv
git checkout --detach origin/agent/multi-qr-yuv
git rev-parse HEAD
git status --short
~~~

## 3. 环境、模型和输入检查

~~~bash
export PYTHON=/mnt/ssd1/z00919662/anaconda3/envs/ultraface/bin/python
export FSD_ROOT=/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
export MODEL_ROOT=$FSD_ROOT/models/qr_fsd_multi_yuv_canonical_v1
export CHECKPOINT=$MODEL_ROOT/qr_fsd_best.pth
export INPUT_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset/from_chenshuo
export OUTPUT_ROOT=/mnt/ssd1/z00919662/qrcode_detection/video_output/from_chenshuo_padded_3x4_canonical_v1

test -f "$CHECKPOINT"
test -d "$INPUT_ROOT"
if [ ! -f "$MODEL_ROOT/CODEAGENT_REPORT.md" ]; then
  echo "WARN: training report not found; continue only with verified best checkpoint"
fi
if [ -e "$OUTPUT_ROOT" ]; then
  echo "STOP: output already exists: $OUTPUT_ROOT"
  exit 2
fi

mapfile -d '' INPUT_VIDEOS < <(
  find "$INPUT_ROOT" -maxdepth 1 -type f -iname '*.mp4' -print0 | sort -z
)
if [ "${#INPUT_VIDEOS[@]}" -ne 2 ]; then
  echo "STOP: expected exactly 2 top-level MP4 files, got ${#INPUT_VIDEOS[@]}"
  printf '%s\n' "${INPUT_VIDEOS[@]}"
  exit 2
fi
printf '%s\n' "${INPUT_VIDEOS[@]}"
~~~

只读记录两个源视频的文件名、SHA256、宽高、FPS、帧数和时长。

## 4. 静态检查和 padding 测试

~~~bash
cd "$PROJECT_ROOT"
"$PYTHON" -m py_compile infer_video.py video_padding.py tests/test_video_padding.py
"$PYTHON" tests/test_video_padding.py
"$PYTHON" -c "import cv2, torch; print(cv2.__version__, torch.__version__)"
~~~

padding 必须满足：

- 先补边，再由模型内部 resize 到 240×320；源画面不做非等比拉伸。
- 输出尺寸严格满足 `width * 4 == height * 3`，并且宽高均为偶数。
- 补边左右或上下居中，两边最多相差 1 像素。
- JSONL 中的检测坐标属于 padded 3:4 画布坐标系。

## 5. 两个视频推理

~~~bash
mkdir -p "$OUTPUT_ROOT"
for input_video in "${INPUT_VIDEOS[@]}"; do
  filename=$(basename "$input_video")
  stem=${filename%.*}
  "$PYTHON" infer_video.py \
    --fsd-repo "$FSD_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --input "$input_video" \
    --padded-input-output "$OUTPUT_ROOT/${stem}_padded_3x4.mp4" \
    --output "$OUTPUT_ROOT/${stem}_qr_detection_padded_3x4.mp4" \
    --pad-to-portrait-3x4 --pad-value 127 \
    --device cuda:0 \
    --score-threshold 0.8 --nms-threshold 0.3 --max-detections 20
done
~~~

`infer_video.py` 内部的 `QRDetector` 固定使用 YUV 模式，不需要额外的 `--input-mode`。
padding 使用中性灰 127，与现有推理 letterbox 的填充值一致。

## 6. 完整性验证

对每个源视频、padded 视频、画框视频确认：

- 三者帧数相同，FPS 和时长一致（允许容器元数据的微小舍入）。
- padded 和画框视频宽高完全一致，且是精确 3:4 偶数尺寸。
- summary JSON 的 `source_size`、`output_size` 和四边 padding 与实际一致。
- JSONL 行数等于源视频帧数，所有记录的 `coordinate_space` 为
  `padded_portrait_3x4`。
- 两个源 MP4 的 SHA256 在推理前后不变。
- 输出文件均非空且可由 OpenCV 从头读到尾。

创建 `$OUTPUT_ROOT/CODEAGENT_INFERENCE_REPORT.md`，报告 commit、checkpoint SHA256、
两个输入及输出路径、源/输出宽高、padding、FPS、帧数、耗时、端到端处理 FPS、总检测数、
有检测的帧数、每帧最大检测数和所有异常。不得修改训练报告或源数据。

## 7. 人工检查

最终明确列出两个画框视频路径和两个未画框 padded 视频路径，要求人工播放确认：画面没有
变形、padding 位置正确、二维码四角与 P0/P1/P2/P3 方向正确、无明显漏检或误检。
