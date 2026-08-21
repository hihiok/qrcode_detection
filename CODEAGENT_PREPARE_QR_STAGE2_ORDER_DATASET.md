# CodeAgent：准备两阶段QR的第二阶段方向数据集

## 1. 任务目标

只下载并运行GitHub中已经准备好的代码，把现有有序角点数据按“一个二维码一个ROI”自动裁剪成第二阶段训练集。

所有第二阶段ROI必须在训练前离线生成并保存为`112×112`正方形JPEG；训练时不得从原图现裁剪。
正方形透视矫正用于恢复二维码平面比例，避免把ROI拉伸为240×320。

第二阶段标签保持：`P0=二维码自身左上角`、`P1=自身右上角`、`P2=自身右下角`、`P3=自身左下角`。

不能从真正只有bbox或无序角点的标签推导二维码自身方向。本项目使用已经确认过P0～P3的BarBeR、
BoofCV、Mendeley和Synth canonical标签生成ROI；第一阶段预测不能替代第二阶段训练真值。

禁止CodeAgent自行写代码、修改代码、commit或push。代码不满足要求时立即停止并报告。

`fore.mp4`和`vrtest.mp4`是锁定的最终评估视频，禁止抽帧、训练、验证、裁剪、hard-negative mining、
阈值选择和人工挑选模型。

## 2. 代理和SSL（clone前必须执行）

代理凭据保存在服务器本地文件，不得把密码复制进GitHub、报告或日志：

```bash
export SOURCE_PROJECT=/mnt/ssd1/z00919662/qrcode_detection
test -s "$SOURCE_PROJECT/proxy.md"
test -s "$SOURCE_PROJECT/CODEAGENT_DISABLE_SSL.md"
export QR_PROXY_URL=$(python - <<'PY'
import re
s=open('/mnt/ssd1/z00919662/qrcode_detection/proxy.md').read()
m=re.search(r'https?://[^\s"`]+', s)
if not m: raise SystemExit('STOP: proxy URL not found in proxy.md')
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

## 3. 下载已准备代码

```bash
export CODE_ROOT=/mnt/ssd1/z00919662/qrcode_detection_two_stage_code
export REPO_URL=https://github.com/hihiok/qrcode_detection.git
export BRANCH=agent/qr-two-stage-ordered-corners-v1
if [ -e "$CODE_ROOT" ]; then
  echo "STOP: $CODE_ROOT already exists; do not overwrite"
  exit 2
fi
git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$CODE_ROOT"
cd "$CODE_ROOT"
git status --short
git rev-parse HEAD
```

工作区必须为空；训练数据和输出不能写进代码仓库。

## 4. 固定路径并检查输入

```bash
export PYTHON=/mnt/ssd1/z00919662/anaconda3/envs/ultraface/bin/python
export SOURCE_DATA=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_train_combined_v3/unified
export STRICT_MANIFEST=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_strict_v2/STRICT_EVAL_MANIFEST.json
export STAGE2_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_stage2_ordered_v1
test -x "$PYTHON"
test -s "$STRICT_MANIFEST"
for split in train val test; do test -s "$SOURCE_DATA/$split/annotations.jsonl"; done
if [ -e "$STAGE2_ROOT" ]; then
  echo "STOP: output already exists: $STAGE2_ROOT"
  exit 2
fi
```

如果实际第一阶段数据根目录不是`$SOURCE_DATA`，只能修改变量指向已冻结、含
`train/val/test/annotations.jsonl`的canonical数据根目录。不得指向两个业务视频或其抽帧目录。

## 5. 代码和输入数据审计

```bash
cd "$CODE_ROOT"
"$PYTHON" -m py_compile *.py tests/*.py
for test_file in tests/test_*.py; do "$PYTHON" "$test_file" || exit 1; done
"$PYTHON" strict_eval_guard.py check \
  --manifest "$STRICT_MANIFEST" --dataset-root "$SOURCE_DATA"
```

必须确认源数据为240×320，每个正实例有四个有序语义角点，P0是二维码自身左上角，三个split无泄漏，
strict-eval leakage为0。若源数据只有bbox或无序polygon，立即停止，不能自动猜测P0。

## 6. 构建第二阶段ROI数据

```bash
cd "$CODE_ROOT"
"$PYTHON" prepare_qr_stage2_dataset.py build \
  --source combined_v3="$SOURCE_DATA" \
  --strict-eval-manifest "$STRICT_MANIFEST" \
  --output "$STAGE2_ROOT" \
  --train-variants 4 --val-variants 1 --test-variants 1 \
  --negative-per-image 1 \
  --margin-min 0.12 --margin-max 0.30 --jitter 0.04 \
  --seed 20260821 | tee "$STAGE2_ROOT.build.log"

"$PYTHON" prepare_qr_stage2_dataset.py verify \
  --manifest "$STAGE2_ROOT/QR_STAGE2_DATASET_MANIFEST.json"
"$PYTHON" strict_eval_guard.py check \
  --manifest "$STRICT_MANIFEST" --dataset-root "$STAGE2_ROOT"
```

训练集每个二维码提前保存4个不同margin/jitter的112×112 ROI，避免只见到完美GT crop；val/test提前
保存确定性112×112 ROI。训练时只读成品ROI，不再执行crop、warp或几何增强。
原负样本保留为空标签，使第二阶段可以拒绝第一阶段误检，而不是强制为所有crop输出方向。

## 7. 必须停止等待人工审核

```bash
ls -lh "$STAGE2_ROOT"/stage2_preview_*.jpg
test -s "$STAGE2_ROOT/stage2_preview_index.json"
```

CodeAgent状态必须为`WAITING_FOR_HUMAN_STAGE2_PREVIEW_REVIEW`，不得自行创建
`APPROVED_STAGE2_DATASET.txt`。

### 需要用户人工执行

用户需要打开全部`stage2_preview_*.jpg`，确认：

1. 每张crop包含目标二维码且留有合理margin；
2. P0/P1/P2/P3顺序正确；
3. 标记贴合真实二维码四角；
4. 没有把相邻二维码当成当前实例；
5. 没有明显空白、错误warp或严重裁断。

若发现问题，记录页码和tile编号，用`stage2_preview_index.json`定位源图；不要批准，也不要让CodeAgent
自行改代码，把问题报告同步给代码作者。

全部通过后，由用户人工创建非空文件：

```text
/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_stage2_ordered_v1/APPROVED_STAGE2_DATASET.txt
```

文件写明审核人、日期和所有preview均通过。此文件是下一条训练指令的硬门槛。

## 8. 本阶段报告

生成`$STAGE2_ROOT/CODEAGENT_STAGE2_DATASET_REPORT.md`，包括commit、输入根目录、各split原图数、
原实例数、正ROI数、负ROI数、manifest SHA-256、strict leakage结果、preview路径及人工审核状态。

人工批准前完成标志：`WAITING_FOR_HUMAN_STAGE2_PREVIEW_REVIEW`。

人工批准后完成标志：`QR_STAGE2_DATASET_READY`。
