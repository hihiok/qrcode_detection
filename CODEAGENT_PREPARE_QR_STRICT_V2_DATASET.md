# CodeAgent 指令一：准备严格隔离的 QR V2 训练数据集

## 1. 唯一目标

从 GitHub 下载已经实现好的代码，构建并冻结 `qr_strict_v2` 数据集。不要修改或编写
Python代码。遇到代码问题立即停止并报告，不得让CodeAgent自行patch。

严格评估规则：

- `fore.mp4`和`vrtest.mp4`只允许计算SHA-256并写入隔离清单。
- 禁止读取/抽帧/截图/查看内容。
- 禁止用于训练、验证、阈值选择、hard-negative mining、背景提取、增强或模型选择。
- 两段视频过去已经看过结果，因此从现在开始把它们称为“锁定业务回归集”，不要宣称是从未看过的blind test。

## 2. 代理与SSL（必须先做）

代理凭据只保存在服务器本地文件，禁止写进GitHub、日志或报告：

```bash
export SOURCE_PROJECT=/mnt/ssd1/z00919662/qrcode_detection
test -f "$SOURCE_PROJECT/proxy.md"
test -f "$SOURCE_PROJECT/CODEAGENT_DISABLE_SSL.md"
sed -n '1,220p' "$SOURCE_PROJECT/proxy.md"
sed -n '1,220p' "$SOURCE_PROJECT/CODEAGENT_DISABLE_SSL.md"
```

按照上述本地文件设置大小写代理环境变量；不得回显值：

```bash
export http_proxy="<从proxy.md读取>"
export https_proxy="<从proxy.md读取>"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org"
```

## 3. 下载我已准备的代码

```bash
export CODE_ROOT=/mnt/ssd1/z00919662/qrcode_detection_strict_v2_data_code
export REPO_URL=https://github.com/hihiok/qrcode_detection.git
export BRANCH=agent/qr-strict-v2-optimization

if [ -e "$CODE_ROOT" ]; then
  echo "STOP: $CODE_ROOT already exists; do not overwrite it"
  exit 2
fi
git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$CODE_ROOT"
cd "$CODE_ROOT"
git status --short
git rev-parse HEAD
```

`git status --short`必须为空。禁止在此仓库写代码、commit或push。

## 4. 固定路径并冻结测试视频

```bash
export PYTHON=/mnt/ssd1/z00919662/anaconda3/envs/ultraface/bin/python
export DATA_BASE=/mnt/ssd1/z00919662/qrcode_detection/dataset
export FORE_VIDEO=$DATA_BASE/from_chenshuo/fore.mp4
export VRTEST_VIDEO=$DATA_BASE/from_chenshuo/vrtest.mp4
export V2_ROOT=$DATA_BASE/qr_strict_v2
export STRICT_MANIFEST=$V2_ROOT/STRICT_EVAL_MANIFEST.json

test -f "$FORE_VIDEO"
test -f "$VRTEST_VIDEO"
if [ -e "$V2_ROOT" ]; then
  echo "STOP: output already exists: $V2_ROOT"
  exit 2
fi
mkdir -p "$V2_ROOT"

"$PYTHON" strict_eval_guard.py freeze \
  --video "$FORE_VIDEO" --video "$VRTEST_VIDEO" \
  --output "$STRICT_MANIFEST"
```

从这一步开始，除`strict_eval_guard.py verify-final-input`外，任何命令都不得接收这两个视频路径。

## 5. 环境和代码测试

```bash
cd "$CODE_ROOT"
"$PYTHON" -V
"$PYTHON" -c "import torch,cv2,numpy,qrcode; print(torch.__version__,cv2.__version__,numpy.__version__)"
"$PYTHON" -m py_compile *.py tests/*.py
for test_file in tests/test_*.py; do "$PYTHON" "$test_file" || exit 1; done
```

必须全部PASS，并确认没有修改FSD结构：输入仍为`[N,3,320,240]`，4720 priors，输出仍为
`confidence [N,4720,2] + ordered_corners [N,4720,8]`，没有bbox head。

## 6. 检查现有真实数据

```bash
export CANONICAL_ROOT=$DATA_BASE/qr_canonical_v1
export BARBER_ROOT=$CANONICAL_ROOT/barber
export BOOFCV_ROOT=$CANONICAL_ROOT/boofcv
export MENDELEY_ROOT=$CANONICAL_ROOT/mendeley

for dataset_root in "$BARBER_ROOT" "$BOOFCV_ROOT" "$MENDELEY_ROOT"; do
  test -f "$dataset_root/train/annotations.jsonl"
  test -f "$dataset_root/val/annotations.jsonl"
  test -f "$dataset_root/test/annotations.jsonl"
  "$PYTHON" validate_qr_dataset.py --data-root "$dataset_root" --visualize 0
done
```

不得重新排序或自动修复已有P0/P1/P2/P3。

## 7. 生成新的高多样性合成集和显式负样本

背景使用现有COCO+ADE图像集；脚本会按源图路径确定性分组，保证同一背景只进入同一个split。

```bash
export BACKGROUND_ROOT=/data/pub1/z00919662/dataset/coco_ADE_12cls
export SYNTH_V2=$V2_ROOT/synth_v2
export NEGATIVE_V2=$V2_ROOT/negative_v2
test -d "$BACKGROUND_ROOT"

"$PYTHON" prepare_qr_dataset.py synthetic \
  --output "$SYNTH_V2" \
  --background-dir "$BACKGROUND_ROOT" \
  --train-count 20000 --val-count 2000 --test-count 2000 \
  --min-qr-side 24 --max-qr-side 200 \
  --max-qrs-per-image 3 --single-qr-probability 0.90 \
  --negative-ratio 0.20 --decoy-probability 0.70 \
  --background-train-ratio 0.80 --background-val-ratio 0.10 \
  --rotate-landscape-cw --seed 20260818

"$PYTHON" prepare_qr_dataset.py negatives \
  --input "$BACKGROUND_ROOT" --output "$NEGATIVE_V2" \
  --max-count 15000 --train-ratio 0.80 --val-ratio 0.10 \
  --decoy-probability 0.40 --seed 20260819

"$PYTHON" validate_qr_dataset.py --data-root "$SYNTH_V2" --visualize 100
"$PYTHON" validate_qr_dataset.py --data-root "$NEGATIVE_V2" --visualize 100
```

合成退化必须包含透视、旋转、尺寸变化、运动/失焦模糊、降采样、JPEG、YUV420、gamma、
低对比度、噪声、反光、摩尔纹以及少量遮挡。正样本约90%为单二维码。

## 8. 用旧模型挖掘hard-negative候选

只能扫描`BACKGROUND_ROOT`，严禁扫描两个锁定视频：

```bash
export FSD_ROOT=/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
export OLD_BEST=$FSD_ROOT/models/qr_fsd_multi_yuv_canonical_v1/qr_fsd_best.pth
export HARDNEG_CANDIDATE=$V2_ROOT/hard_negative_candidate
test -f "$OLD_BEST"

"$PYTHON" mine_hard_negatives.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$OLD_BEST" \
  --input "$BACKGROUND_ROOT" --output "$HARDNEG_CANDIDATE" \
  --strict-eval-manifest "$STRICT_MANIFEST" \
  --device cuda:0 --score-threshold 0.35 \
  --max-candidates 1000 --preview-count 1000
```

脚本会先排除OpenCV能够检测到二维码的图片，但仍然必须人工检查：

- `$HARDNEG_CANDIDATE/hard_negative_preview_*.jpg`
- `$HARDNEG_CANDIDATE/HARD_NEGATIVE_REVIEW.json`

### 必须暂停的人工步骤

此处CodeAgent必须停止并告诉用户预览路径，不能自行创建`APPROVED_BY_HUMAN.txt`。

用户需要检查所有预览页：

- 如果全部确实没有二维码，用户人工创建一个非空文件：
  `$HARDNEG_CANDIDATE/APPROVED_BY_HUMAN.txt`，写明检查日期和“1000 candidates contain no real QR”。
- 如果发现真实二维码，用户把对应`source`路径逐行写入`hard_negative_exclude.txt`，然后让CodeAgent
  使用`--exclude-source-file`重新生成到新的`hard_negative_reviewed`目录，再次人工检查。不得直接删除图片，
  因为这样会造成annotations与文件不一致。

重新生成命令为：

```bash
export HARDNEG_REVIEWED=$V2_ROOT/hard_negative_reviewed
test -s "$V2_ROOT/hard_negative_exclude.txt"
test ! -e "$HARDNEG_REVIEWED"
"$PYTHON" mine_hard_negatives.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$OLD_BEST" \
  --input "$BACKGROUND_ROOT" --output "$HARDNEG_REVIEWED" \
  --strict-eval-manifest "$STRICT_MANIFEST" \
  --device cuda:0 --score-threshold 0.35 \
  --max-candidates 1000 --preview-count 1000 \
  --exclude-source-file "$V2_ROOT/hard_negative_exclude.txt"
```

重新检查并批准后，把第9节的`HARDNEG_FINAL`改为`$HARDNEG_REVIEWED`。

没有非空人工批准文件时，后续manifest构建会强制失败。

## 9. 人工批准后冻结V2 manifest

只有在用户明确完成上一步后才能继续：

```bash
export HARDNEG_FINAL=$HARDNEG_CANDIDATE
test -s "$HARDNEG_FINAL/APPROVED_BY_HUMAN.txt"
export DATASET_MANIFEST=$V2_ROOT/QR_DATASET_V2_MANIFEST.json

"$PYTHON" dataset_v2_manifest.py build \
  --source barber="$BARBER_ROOT" \
  --source boofcv="$BOOFCV_ROOT" \
  --source mendeley="$MENDELEY_ROOT" \
  --source synth="$SYNTH_V2" \
  --source negative="$NEGATIVE_V2" \
  --source hard_negative="$HARDNEG_FINAL" \
  --weight barber=0.16 --weight boofcv=0.10 --weight mendeley=0.09 \
  --weight synth=0.35 --weight negative=0.20 --weight hard_negative=0.10 \
  --kind barber=real --kind boofcv=real --kind mendeley=real \
  --kind synth=synthetic --kind negative=negative \
  --kind hard_negative=hard_negative \
  --strict-eval-manifest "$STRICT_MANIFEST" \
  --output "$DATASET_MANIFEST"

"$PYTHON" dataset_v2_manifest.py verify --manifest "$DATASET_MANIFEST"
"$PYTHON" audit_qr_training_pipeline.py \
  --manifest "$DATASET_MANIFEST" --split train \
  --max-images-per-source 3000 --preview-per-source 12 \
  --output-dir "$V2_ROOT/audit"
```

人工批准后manifest会冻结每个annotations文件及全部图像内容的聚合SHA-256、图像/实例/负样本数量、
数据源权重和hard-negative批准文件SHA-256。后续任何变化都会报错。

## 10. 最终报告

生成`$V2_ROOT/CODEAGENT_DATASET_REPORT.md`，必须报告：

- Git commit、Python/torch/OpenCV版本；
- 两个锁定视频SHA-256，但不得截图、抽帧或描述内容；
- 每个source和split的图片数、QR实例数、负样本数；
- QR尺寸分桶、每图正anchor数量；
- synthetic退化配置、单二维码比例；
- hard-negative人工批准状态；
- manifest、audit JSON和预览路径；
- 明确写出`fore.mp4/vrtest.mp4 were not used for data or tuning`；
- 所有失败、跳过项和仍需人工完成的事项。

若尚未取得人工批准，报告状态必须是`WAITING_FOR_HUMAN_HARD_NEGATIVE_REVIEW`，不得伪装完成。
