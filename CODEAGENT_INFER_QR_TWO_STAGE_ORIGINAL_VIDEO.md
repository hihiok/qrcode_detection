# CodeAgent：两阶段二维码业务视频推理，输出原分辨率 MP4 和角点

## 1. 任务范围与固定结构

只下载 GitHub 已准备好的新增推理代码，使用已有 Stage 1、Stage 2 checkpoint 推理业务 MP4。

- Stage 1 模型输入固定为 `240×320`，即 `N×3×320×240`。
- Stage 2 模型输入固定为 `112×112`，即 `N×3×112×112`。
- 不修改网络、checkpoint、训练代码、数据集或已有推理脚本。
- 内部可补边为 `3:4`，但最终可视化 MP4 宽高必须与原始 MP4 完全一致。
- P0、P1、P2、P3 坐标必须是原始视频分辨率下的像素坐标，不得保留 padding 偏移。
- 每个输入视频生成原分辨率可视化 MP4、逐帧 JSONL、坐标 CSV 和 summary JSON。
- 禁止 CodeAgent 自行编写、修改、提交或上传代码；代码不满足要求时立即停止并向用户报告。
- `fore.mp4`、`vrtest.mp4` 是严格最终评估视频；只允许使用已冻结 checkpoint、阈值和 crop margin，
  不得用于训练、调参、阈值搜索、挑选模型或 hard-negative mining。

## 2. 代理和 SSL：任何 clone/fetch 之前必须执行

代理凭据只从服务器本地 `/mnt/ssd1/z00919662/qrcode_detection/proxy.md` 读取，不得写入公开 GitHub、
终端输出或报告；同时检查 `/mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md`。

```bash
export SOURCE_PROJECT=/mnt/ssd1/z00919662/qrcode_detection
test -s "$SOURCE_PROJECT/proxy.md"
test -s "$SOURCE_PROJECT/CODEAGENT_DISABLE_SSL.md"
export QR_PROXY_URL=$(python - <<'PY'
import re
with open('/mnt/ssd1/z00919662/qrcode_detection/proxy.md') as handle:
    source = handle.read()
match = re.search(r'https?://[^\s"`]+', source)
if not match:
    raise SystemExit('STOP: proxy URL not found in local proxy.md')
print(match.group(0))
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

禁止开启 `set -x`，禁止打印代理 URL，禁止将代理密码写进结果文件。

## 3. 获取现有分支最新代码

```bash
export CODE_ROOT=/mnt/ssd1/z00919662/qrcode_detection_two_stage_original_video_code
export REPO_URL=https://github.com/hihiok/qrcode_detection.git
export BRANCH=agent/qr-two-stage-ordered-corners-v1

if [ -d "$CODE_ROOT/.git" ]; then
  cd "$CODE_ROOT"
  test -z "$(git status --porcelain)" || {
    echo 'STOP: existing checkout has local changes; do not overwrite'
    exit 2
  }
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
elif [ -e "$CODE_ROOT" ]; then
  echo "STOP: existing non-git path: $CODE_ROOT"
  exit 2
else
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$CODE_ROOT"
  cd "$CODE_ROOT"
fi

git rev-parse HEAD
test -s infer_qr_two_stage_original_video.py
test -s qr_business_video_geometry.py
test -s tests/test_qr_business_video_geometry.py
```

## 4. 固定环境、模型和视频路径

```bash
export PYTHON=/mnt/ssd1/z00919662/anaconda3/envs/ultraface/bin/python
export FSD_ROOT=/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
export MODEL_ROOT=$FSD_ROOT/models/qr_fsd_two_stage_ordered_v1
export STAGE1_BEST=$MODEL_ROOT/stage1_geometry/qr_fsd_best.pth
export STAGE2_BEST=$MODEL_ROOT/stage2_semantic/qr_fsd_best.pth
export FROZEN_CONFIG=$MODEL_ROOT/FROZEN_TWO_STAGE_CONFIG.json
export STRICT_MANIFEST=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_strict_v2/STRICT_EVAL_MANIFEST.json
export FORE=/mnt/ssd1/z00919662/qrcode_detection/dataset/from_chenshuo/fore.mp4
export VRTEST=/mnt/ssd1/z00919662/qrcode_detection/dataset/from_chenshuo/vrtest.mp4
export VIDEO_OUT=/mnt/ssd1/z00919662/qrcode_detection/video_output/two_stage_original_resolution_v1

test -x "$PYTHON"
test -d "$FSD_ROOT"
test -s "$STAGE1_BEST"
test -s "$STAGE2_BEST"
test -s "$FROZEN_CONFIG"
test -s "$STRICT_MANIFEST"
test -s "$FORE"
test -s "$VRTEST"
mkdir -p "$VIDEO_OUT"
```

若实际 checkpoint 位于其他目录，只能将 `STAGE1_BEST`、`STAGE2_BEST`、`FROZEN_CONFIG` 指向已经训练
完成且冻结的真实文件；不得自行训练、猜测权重、调整阈值或修改代码。任何文件缺失都必须停止并报告准确路径。

## 5. 静态检查、坐标测试和严格评估验证

```bash
cd "$CODE_ROOT"
"$PYTHON" -m py_compile \
  infer_qr_two_stage_original_video.py \
  qr_business_video_geometry.py \
  tests/test_qr_business_video_geometry.py

"$PYTHON" tests/test_qr_business_video_geometry.py

for video in "$FORE" "$VRTEST"; do
  "$PYTHON" strict_eval_guard.py verify-final-input \
    --manifest "$STRICT_MANIFEST" --video "$video" || exit 1
done
```

必须看到 `PASS: original-resolution video coordinate recovery`。测试覆盖：

- `480×408 → 480×640` 内部补边时 `top=116` 正确扣除；
- 水平 padding 和图像边界 clipping；
- 落在纯 padding 区域的误检被丢弃；
- 无 padding 时原始坐标不发生变化。

## 6. 推理 fore.mp4

```bash
cd "$CODE_ROOT"

"$PYTHON" infer_qr_two_stage_original_video.py \
  --fsd-repo "$FSD_ROOT" \
  --stage1-checkpoint "$STAGE1_BEST" \
  --stage2-checkpoint "$STAGE2_BEST" \
  --input "$FORE" \
  --output "$VIDEO_OUT/fore_original_resolution.mp4" \
  --frozen-config "$FROZEN_CONFIG" \
  --device cuda:0 \
  --pad-to-portrait-3x4 \
  --pad-value 127 \
  --strict-eval-manifest "$STRICT_MANIFEST" \
  | tee "$VIDEO_OUT/fore_original_resolution.log"
```

输出文件：

```text
fore_original_resolution.mp4
fore_original_resolution_original_coords.jsonl
fore_original_resolution_original_coords.csv
fore_original_resolution_summary.json
```

## 7. 推理 vrtest.mp4

```bash
cd "$CODE_ROOT"

"$PYTHON" infer_qr_two_stage_original_video.py \
  --fsd-repo "$FSD_ROOT" \
  --stage1-checkpoint "$STAGE1_BEST" \
  --stage2-checkpoint "$STAGE2_BEST" \
  --input "$VRTEST" \
  --output "$VIDEO_OUT/vrtest_original_resolution.mp4" \
  --frozen-config "$FROZEN_CONFIG" \
  --device cuda:0 \
  --pad-to-portrait-3x4 \
  --pad-value 127 \
  --strict-eval-manifest "$STRICT_MANIFEST" \
  | tee "$VIDEO_OUT/vrtest_original_resolution.log"
```

## 8. 验收

两个视频分别确认：

1. 日志出现 `TWO_STAGE_ARCHITECTURE_MATCH`。
2. Stage 1 日志显示 `NCHW=(1, 3, 320, 240)` 和 4720 priors。
3. Stage 2 日志显示 `NCHW=(1, 3, 112, 112)` 和 779 priors。
4. 日志出现 `ORIGINAL_RESOLUTION_TWO_STAGE_INFERENCE_PASS`。
5. summary JSON 的 `source_size == output_size`。
6. summary JSON 的 `stage1_model_input_size == [240, 320]`。
7. summary JSON 的 `stage2_model_input_size == [112, 112]`。
8. JSONL 每一帧都有记录；`coordinate_space == "original_video_pixels"`。
9. 每个 P0～P3 满足 `0 ≤ x < 原始宽度`、`0 ≤ y < 原始高度`。
10. CSV 包含 `p0_x,p0_y,p1_x,p1_y,p2_x,p2_y,p3_x,p3_y`。
11. 可视化 MP4 显示原始画面、二维码四边形、P0～P3 和实际像素坐标，不包含 padding 边框。

最终向用户报告 Git commit、两个视频输入/输出分辨率、帧数、FPS、检测数量、完整输出路径和异常。

最终完成标志：`QR_TWO_STAGE_ORIGINAL_RESOLUTION_VIDEO_COMPLETE`。
