# CodeAgent：统一四个二维码数据集为 canonical JSONL

## 1. 目标与禁止事项

将 BarBeR、BoofCV、Mendeley 和旧合成集转换为唯一格式
`qr_ordered_corners_v1`，输出到新目录。源数据集只读，禁止覆盖、移动、重排角点或重新生成。

本任务只做转换、验证和预览，不训练。转换结果人工确认后再开始正式训练。

## 2. 代理与 SSL

执行任何 git/pip/curl 网络操作前，先读取服务器配置：

~~~bash
test -f /mnt/ssd1/z00919662/qrcode_detection/proxy.md
test -f /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md
sed -n '1,220p' /mnt/ssd1/z00919662/qrcode_detection/proxy.md
sed -n '1,220p' /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md
~~~

按服务器配置设置代理，并在 git fetch/clone 前关闭 Git SSL 校验：

~~~bash
export http_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
git config --global https.proxy "http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
git config --global http.sslVerify false
~~~

最终报告说明是否使用代理和 SSL bypass，但不得回显代理账号或密码。

## 3. 更新现有干净 worktree

~~~bash
export PROJECT_ROOT=/mnt/ssd1/z00919662/qrcode_detection_multi_qr_yuv
cd "$PROJECT_ROOT"
git status --short
~~~

如果输出非空，立即停止；不得 stash、clean、reset 或覆盖文件。工作区干净时：

~~~bash
git fetch origin agent/multi-qr-yuv
git checkout --detach origin/agent/multi-qr-yuv
git rev-parse HEAD
git status --short
~~~

## 4. 路径

~~~bash
export SOURCE_BASE=/mnt/ssd1/z00919662/qrcode_detection/dataset
export BARBER_SOURCE=$SOURCE_BASE/barber_qr_accepted_1027_240x320
export BOOFCV_SOURCE=$SOURCE_BASE/boofcv_qr_multi_240x320_rotation_validated
export MENDELEY_SOURCE=$SOURCE_BASE/mendeley_qr_multi_240x320_rotation_validated
export SYNTH_SOURCE=$SOURCE_BASE/qr_full_data
export CANONICAL_ROOT=$SOURCE_BASE/qr_canonical_v1
~~~

四个 source 必须存在；`$CANONICAL_ROOT` 必须不存在。若已经存在，停止并报告，禁止删除或覆盖。

## 5. 静态检查和单元测试

沿用 ultraface 环境，不升级依赖：

~~~bash
export PYTHON=/mnt/ssd1/z00919662/anaconda3/envs/ultraface/bin/python
cd "$PROJECT_ROOT"
"$PYTHON" -m py_compile qr_schema.py canonicalize_qr_datasets.py \
  qr_dataset.py validate_qr_dataset.py tests/test_qr_schema.py \
  tests/test_canonicalize_qr_datasets.py tests/test_qr_dataset.py
"$PYTHON" tests/test_qr_schema.py
"$PYTHON" tests/test_canonicalize_qr_datasets.py
"$PYTHON" tests/test_qr_dataset.py
~~~

## 6. 转换

~~~bash
"$PYTHON" canonicalize_qr_datasets.py \
  --dataset barber="$BARBER_SOURCE" \
  --dataset boofcv="$BOOFCV_SOURCE" \
  --dataset mendeley="$MENDELEY_SOURCE" \
  --dataset synth="$SYNTH_SOURCE" \
  --deduplicate-identical-images \
  --conflicting-duplicate-keeper \
    barber:val:images/barber_945dae373601f3c3.jpg \
  --output-root "$CANONICAL_ROOT"
~~~

转换器必须：

- 保留原始像素角点和 P0/P1/P2/P3 顺序，不重新计算或排序。
- 把 BarBeR `instances`、BoofCV/Mendeley `objects`、SYNTH 顶层 `corners`
  统一成 `instances[]`。
- 对声明了 `label_file` 的数据逐坐标交叉检查 JSON 与 TXT。
- 检查 `num_qrcodes`、角点范围、退化四边形、图片存在性、组级 split 泄漏和精确图片重复。
- `--deduplicate-identical-images` 不是跳过检查：仅当 SHA256 完全相同且 canonical
  标签逐字段完全相同时才自动去重。保留优先级固定为
  `test > val > train`，同一 split 内以图片路径字典序决定。
- 标签冲突默认立即失败。`--conflicting-duplicate-keeper` 是逐个重复组的显式人工决定：
  参数必须精确匹配重复组中的一个现有 split/图片路径；路径写错、未匹配、匹配不唯一，
  或出现任何其他未授权的冲突重复，都必须失败。
- 通过软链接引用源 `images/labels`，不复制或修改源文件。
- 任一错误时删除自身临时 staging 目录并失败；绝不留下可被误用的正式输出目录。

已知 BarBeR 唯一跨 split 重复具有完全相同的图片内容，但两边各有一个 QR，角点标签
存在约 0.01～0.2 像素差异。已经明确决定保留 val 标签。预期转换报告必须显示：

- 保留 val：`barber_945dae373601f3c3.jpg`
- canonical train 排除：`barber_cae21b30f66a5651.jpg`
- `labels_identical: false`
- `explicit_conflict_resolution: true`
- train 的 `dropped_exact_duplicates: 1`

这里的“排除”只影响新 canonical JSONL；源图片和源 annotations 不得删除或修改。

## 7. 验证及人工预览

~~~bash
for dataset_name in barber boofcv mendeley synth; do
  "$PYTHON" validate_qr_dataset.py \
    --data-root "$CANONICAL_ROOT/$dataset_name" --visualize 50
done
~~~

必须确认 BoofCV 和 Mendeley 的所有 split 不再是 0 instances，并检查：

~~~text
$CANONICAL_ROOT/conversion_report.json
$CANONICAL_ROOT/<dataset>/validation_report.json
$CANONICAL_ROOT/<dataset>/validation_preview/
~~~

需要人工查看四个 `validation_preview`，重点确认 P0/P1/P2/P3 的二维码自身方向。

## 8. 最终报告

报告实际 commit、源/输出路径、每个数据集每个 split 的图片/实例/负样本/实例直方图、
JSON-TXT核对数量、group_key检查、精确重复检查、源 annotations SHA256 未变化、所有测试结果，
`duplicate_resolutions` 的 keeper/dropped 明细，以及需要人工检查的预览路径。
本任务结束后不得自行开始训练。
