# CodeAgent：导出 BarBeR 1027 张已接受数据与完整预览

## 目标

只保留已经确认角点语义顺序的两类数据：

- V3 人工 gold：799 张、915 个实例；
- ZXing 全图 ZA 自动通过：228 张、229 个实例。

最终导出 1027 张图片、1144 个二维码实例。明确按整图丢弃全部 ZB/ZM 图片：192
张、246 个实例。被丢弃的图片中有237个ZB/ZM实例，以及与它们同图的9个ZA实例。
Audit报告中的 `ZA_instances=238` 是实例级统计，不等于最终保留的ZA实例数：只有
位于全图ZA图片中的229个实例被保留。输出必须是新的自包含数据集，包含
`images/`、`labels/`、重建后的
`annotations.jsonl`、1027 张单图预览和覆盖全部图片的分页总览。

不得把原数据集旧 TXT 当作角点来源。导出标签只能来自已通过 validation 的
`combined_proposed`。

## 代理与 SSL

在任何 `git clone`、`git fetch`、`git pull` 或 `pip install` 前执行：

```bash
export http_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export https_proxy="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
export HTTPS_PROXY="http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080"
git config --global http.proxy http://z00919662:Zzhs12345%21@proxy.server.com:8080
git config --global https.proxy http://z00919662:Zzhs12345%21@proxy.server.com:8080
git config --global https.proxy https://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080
git config --global http.proxy http://z00919662:Zzhs12345%21@proxyhk.huawei.com:8080
git config --global http.sslVerify false
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"
```

不要在控制台或最终报告中打印代理环境变量或回显代理密码。

## 固定路径

```bash
export QR_REPO=/mnt/ssd1/z00919662/qrcode_detection
export QR_RUNNER=/mnt/ssd1/z00919662/qrcode_detection_barber_accepted_1027_runner
export QR_DATA=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated
export QR_WORK=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v3_geometry_work_v4
export QR_OUTPUT=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_accepted_1027_240x320
export QR_BRANCH=agent/barber-vgg-geometry-v3
export QR_COMMIT="${QR_COMMIT:?set the exact commit supplied by ChatGPT}"
```

`QR_OUTPUT` 必须是全新路径。禁止删除或覆盖已有目录。

## 禁止项

- 不要修改、删除或覆盖 `$QR_DATA`、`$QR_WORK`。
- 不要修改代码、测试、MD、门限或报告。
- 不要创建临时 Python 脚本。
- 不要使用旧 dataset TXT 生成新角点。
- 不要将 ZB/ZM 图片或标签复制进新数据集。
- 不要 finalize、apply、commit、push 或 merge。
- 不要读取或输出二维码 payload。

如果发现必须改代码，或者 runner 出现任何源码/脚本差异：

1. 立即停止导出；
2. 保存 `git status --short`、`git diff --binary` 和未跟踪文件清单；
3. 通过聊天附件、Google Drive 或专用临时 GitHub 分支同步给 ChatGPT；
4. 等待 ChatGPT 发布新提交；
5. CodeAgent 不得自行修复。

## 1. 获取固定版本 runner

```bash
test ! -e "$QR_RUNNER" || {
  echo "ERROR: QR_RUNNER already exists; stop"
  exit 2
}

git clone --single-branch --branch "$QR_BRANCH" \
  https://github.com/hihiok/qrcode_detection.git "$QR_RUNNER"
cd "$QR_RUNNER"
git checkout --detach "$QR_COMMIT"
test "$(git rev-parse HEAD)" = "$QR_COMMIT"
test -z "$(git status --short)"
```

## 2. 环境与测试

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/ssd1/z00919662/qrcode_detection/.tools/zxingcpp_py311
cd "$QR_RUNNER"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m pip install --no-cache-dir \
  -r barber_label_repair_zxingcpp_v2/requirements.txt
python -m pytest -q barber_label_repair_zxingcpp_v2/tests
```

必须至少 21 项测试全部通过，其中包括：

- 不复制旧 dataset TXT 顺序；
- ZB/ZM 不能进入 accepted 输出；
- 输出存在时拒绝覆盖；
- `annotations.jsonl` 使用 `combined_proposed` 的 P0-P3；
- 每张已接受图片都有单图预览。

## 3. 输入报告门禁

```bash
test -f "$QR_WORK/recovery_report.json"
test -f "$QR_WORK/validation_report.json"
test -f "$QR_WORK/recovery_failures.jsonl"
test -d "$QR_WORK/combined_proposed"
test ! -e "$QR_OUTPUT"
```

导出程序会再次强制检查：

- `validation_report.passed=true`；
- `vgg_gold_cross_validation_passed=true`；
- `old_txt_geometry_used=false`；
- `combined_proposed_images=1027`；
- `combined_proposed_instances=1144`；
- ZB+ZM 恰好192张；
- V3 gold保留915实例、全图ZA保留229实例；
- 整图丢弃246实例，其中237个为ZB/ZM、9个为同图ZA；
- `recovery_failures.jsonl` 中的任何 image_id 都不能进入 accepted 集。

## 4. 导出新数据集和1027张预览

```bash
set -o pipefail
python -m barber_label_repair_zxingcpp_v2.accepted_export \
  --dataset "$QR_DATA" \
  --work "$QR_WORK" \
  --output "$QR_OUTPUT" \
  --expected-images 1027 \
  --expected-instances 1144 \
  --expected-dropped-images 192 \
  --expected-v3-images 799 \
  --expected-za-images 228 \
  --expected-gold-instances 915 \
  --expected-accepted-za-instances 229 \
  --expected-dropped-instances 246 \
  --expected-dropped-non-za-instances 237 \
  --expected-dropped-embedded-za-instances 9 \
  --page-size 20 \
  --columns 4 \
  2>&1 | tee "$QR_OUTPUT.export_console.log"
export_rc=${PIPESTATUS[0]}
test "$export_rc" -eq 0 || {
  echo "ERROR: accepted export failed"
  exit "$export_rc"
}
```

输出结构：

```text
$QR_OUTPUT/
  train|val|test/
    images/
    labels/
    annotations.jsonl
  accepted_manifest.jsonl
  accepted_export_report.json
  preview/
    preview_report.json
    preview_manifest.jsonl
    per_image/train|val|test/*.jpg
    pages/accepted_page_001.jpg ... accepted_page_052.jpg
```

## 5. 强制验收

```bash
python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["QR_OUTPUT"])
report = json.loads((root / "accepted_export_report.json").read_text())
preview = json.loads((root / "preview" / "preview_report.json").read_text())

assert report["passed"] is True, report
assert report["images"] == 1027, report
assert report["instances"] == 1144, report
assert report["sources"] == {"V3_gold": 799, "ZXing_ZA": 228}, report
assert report["source_instances"] == {"V3_gold": 915, "ZXing_ZA": 229}, report
assert report["dropped_images"] == 192, report
assert report["dropped_instances"] == 246, report
assert report["dropped_non_za_instances"] == 237, report
assert report["dropped_embedded_za_instances"] == 9, report
assert report["old_dataset_txt_used_as_geometry"] is False, report
assert report["original_inputs_unchanged"] is True, report
assert preview["passed"] is True, preview
assert preview["images"] == 1027, preview
assert preview["instances"] == 1144, preview
assert preview["per_image_previews"] == 1027, preview
assert preview["pages"] == 52, preview

images = sum(1 for split in ("train", "val", "test")
             for p in (root / split / "images").iterdir() if p.is_file())
labels = sum(1 for split in ("train", "val", "test")
             for p in (root / split / "labels").glob("*.txt"))
annotations = sum(len((root / split / "annotations.jsonl").read_text().splitlines())
                  for split in ("train", "val", "test"))
per_image = len(list((root / "preview" / "per_image").rglob("*.jpg")))
pages = len(list((root / "preview" / "pages").glob("accepted_page_*.jpg")))
assert (images, labels, annotations, per_image, pages) == (1027, 1027, 1027, 1027, 52)
print(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps(preview, indent=2, sort_keys=True))
PY
```

必须额外确认：

```bash
test -z "$(git status --short)"
git diff --exit-code
```

## 6. 最终报告

报告：

- branch、精确 commit、测试数量；
- 输出路径；
- 1027张/1144实例；
- V3 799张/915实例、全图ZA 228张/229实例；
- 丢弃 ZB/ZM 图片192张/246实例，其中237个ZB/ZM实例、9个同图ZA实例；
- train/val/test 各自图片和实例数；
- `annotations.jsonl`、labels、images 各1027；
- 单图预览1027张、分页总览52页；
- 第一页与最后一页完整路径；
- 原数据、原work、Git仓库均未改变。

本轮停止在导出完成。不要训练、finalize、apply 或修改任何标签。
