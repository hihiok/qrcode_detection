# CodeAgent：BarBeR VGG Geometry V3 重建与 420 张重新审计

## 目标

废弃 ZXing-C++ V2 对剩余 420 张生成的审计结果、ZA/ZB/ZM 统计、21 张预览页和
decision template。保留 V3 人工确认的 799 张/915 实例以及已经通过的 ZXing 语义
校准。

本版本必须从原始 BarBeR VIA/VGG polygon 重建几何：

1. 用处理后数据集的 `annotations.jsonl` 找回原始 BarBeR 图片；
2. 通过渲染与 SSIM 选择唯一 source；
3. 对横图执行顺时针 90° 旋转；
4. 使用与 V3 相同的等比例 resize 和居中 padding；
5. 对图片和 VGG polygon 应用完全相同的变换；
6. 用 799 张/915 实例 V3 gold 做逐顶点交叉验证；
7. 交叉验证全部通过后，才允许重新审计剩余 420 张。

旧的 `train/val/test/labels/*.txt` 只参加不可变性哈希，不得用于 audit、review、
finalize 的 polygon、bbox 或坐标来源。

## 固定路径

```bash
export QR_REPO=/mnt/ssd1/z00919662/qrcode_detection
export QR_RUNNER=/mnt/ssd1/z00919662/qrcode_detection_barber_vgg_geometry_v3_runner
export QR_DATA=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated
export QR_BARBER=/mnt/ssd1/z00919662/qrcode_detection/dataset/BarBeR
export QR_V3=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_v3_work
export QR_V2_WORK=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v2_work
export QR_WORK=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v3_geometry_work
export QR_BRANCH=agent/barber-vgg-geometry-v3
```

`QR_WORK` 必须是全新目录。不要删除、覆盖或复用旧 V2 work。

## 禁止项

- 不要修改代码或门限。
- 不要使用 `--allow-count-mismatch`。
- 不要修改 `$QR_DATA`、`$QR_BARBER`、`$QR_V3` 或 `$QR_V2_WORK`。
- 不要读取旧 TXT 作为几何真值。
- 不要把 ZXing position 写入标签。
- 不要输出二维码 payload；只能保留 SHA-256。
- 不要填写人工 decision，不要 finalize，不要 apply。
- 不要 commit、push 或 merge。

## 1. 获取独立 runner

```bash
test ! -e "$QR_RUNNER" || {
  echo "ERROR: QR_RUNNER already exists; stop and report"
  exit 2
}
git clone --single-branch --branch "$QR_BRANCH" \
  https://github.com/hihiok/qrcode_detection.git "$QR_RUNNER"
cd "$QR_RUNNER"
git status --short
git log -1 --oneline
```

工作区必须干净。

## 2. 环境与依赖

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/ssd1/z00919662/qrcode_detection/.tools/zxingcpp_py311
cd "$QR_RUNNER"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pip install --no-cache-dir \
  -r barber_label_repair_zxingcpp_v2/requirements.txt
python - <<'PY'
import sys, zxingcpp
from PIL import Image
import numpy
print("python", sys.version)
print("zxingcpp", getattr(zxingcpp, "__version__", "imported"))
print("Pillow", Image.__version__ if hasattr(Image, "__version__") else "imported")
print("numpy", numpy.__version__)
PY
```

如需代理或 SSL 设置，只能读取 `$QR_REPO/CODEAGENT_DISABLE_SSL.md` 和
`$QR_REPO/proxy.md`。

## 3. 静态安全检查与测试

```bash
test -f barber_label_repair_zxingcpp_v2/vgg_geometry.py
test -f CODEAGENT_RUN_BARBER_ZXINGCPP_V3_GEOMETRY.md

if rg -n 'parse_label\(rec\.label_path|shutil\.copy2\(rec\.label_path' \
  barber_label_repair_zxingcpp_v2/pipeline.py \
  barber_label_repair_zxingcpp_v2/review.py; then
  echo "ERROR: old TXT is still used as geometry"
  exit 2
fi

if rg -n 'result\.text|import cv2|from cv2' \
  barber_label_repair_zxingcpp_v2 --glob '*.py' --glob '!tests/**'; then
  echo "ERROR: forbidden production pattern"
  exit 2
fi

python -m pytest -q barber_label_repair_zxingcpp_v2/tests
```

必须至少 16 个测试全部通过，包括：

- 原始 VGG 五点闭合 polygon 安全归一为四点；
- landscape 顺时针旋转、resize、padding 的坐标变换；
- V3 gold 与 VGG 顶点集合交叉验证；
- 真实 zxing-cpp 四旋转检测；
- 旧 TXT token 不会进入新的 recovered label；
- 退化 VGG 实例不会中断整批。

## 4. Geometry Doctor：重建 1219 张 VGG 几何

```bash
test ! -e "$QR_WORK" || {
  echo "ERROR: QR_WORK must be new; stop and report"
  exit 2
}
mkdir -p "$QR_WORK"

set -o pipefail
python -m barber_label_repair_zxingcpp_v2.cli doctor \
  --dataset "$QR_DATA" \
  --barber-root "$QR_BARBER" \
  --v3-work "$QR_V3" \
  --work "$QR_WORK" \
  2>&1 | tee "$QR_WORK/geometry_doctor_console.log"
doctor_rc=${PIPESTATUS[0]}
test "$doctor_rc" -eq 0 || {
  echo "ERROR: geometry doctor failed; stop before audit"
  exit "$doctor_rc"
}
```

强制检查：

```bash
python - <<'PY'
import json, os
from pathlib import Path
w = Path(os.environ["QR_WORK"])
doctor = json.loads((w / "doctor_report.json").read_text())
geometry = json.loads((w / "vgg_geometry_report.json").read_text())
gold = json.loads((w / "vgg_gold_cross_validation_report.json").read_text())
assert doctor["passed"] is True, doctor
assert doctor["dataset_images"] == 1219, doctor
assert doctor["V3_gold_images"] == 799, doctor
assert doctor["failed_images"] == 420, doctor
assert doctor["old_txt_geometry_used"] is False, doctor
assert gold["passed"] is True, gold
assert gold["gold_images"] == 799, gold
assert gold["matched_instances"] == 915, gold
assert gold["error_count"] == 0, gold
assert gold["maximum_vertex_error_px"] <= gold["tolerance_px"], gold
print(json.dumps({"doctor": doctor, "geometry": geometry, "gold": gold},
                 indent=2, sort_keys=True))
PY
```

只要 799/915 有一个未通过，立即停止；不得 audit、不得修改门限、不得回退旧 TXT。

必须保留并报告：

- `vgg_geometry_manifest.jsonl`
- `vgg_geometry_report.json`
- `vgg_geometry_failures.jsonl`
- `vgg_parse_errors.jsonl`
- `vgg_gold_cross_validation_report.json`
- `failed_420_manifest.jsonl`

## 5. 复用旧 V2 的已通过 calibration，并重新审计 420 张

不要重新校准。新版会严格检查旧报告后只复制 calibration report 和 mapping 到新
work；不会复制旧 audit 或旧 review pack。

```bash
set -o pipefail
python -m barber_label_repair_zxingcpp_v2.cli audit \
  --dataset "$QR_DATA" \
  --barber-root "$QR_BARBER" \
  --v3-work "$QR_V3" \
  --work "$QR_WORK" \
  --reuse-calibration-work "$QR_V2_WORK" \
  2>&1 | tee "$QR_WORK/audit_console.log"
audit_rc=${PIPESTATUS[0]}
test "$audit_rc" -eq 0 || {
  echo "ERROR: audit failed; preserve all outputs and report traceback"
  exit "$audit_rc"
}
```

## 6. 强制验收 audit

```bash
python - <<'PY'
import json, os
from pathlib import Path
w = Path(os.environ["QR_WORK"])
recovery = json.loads((w / "recovery_report.json").read_text())
validation = json.loads((w / "validation_report.json").read_text())
reuse = json.loads((w / "calibration_reuse_report.json").read_text())
assert recovery["input_failed_images"] == 420, recovery
assert recovery["ZA_images"] + recovery["ZB_images"] + recovery["ZM_images"] == 420
assert recovery["combined_proposed_images"] == 799 + recovery["ZA_images"]
assert recovery["geometry_source"] == "barber_vgg_manual_polygon"
assert recovery["old_txt_geometry_used"] is False
assert validation["passed"] is True, validation
assert validation["vgg_gold_cross_validation_passed"] is True
assert validation["polygon_vertex_token_equality"] is True
assert validation["old_txt_geometry_used"] is False
assert reuse["passed"] is True, reuse
print(json.dumps({"calibration_reuse": reuse, "recovery": recovery,
                  "validation": validation}, indent=2, sort_keys=True))
PY
```

注意：`input_failed_instances` 现在包含 VGG 无法解析时的显式 ZM 占位项；同时必须
单独报告 `auditable_vgg_instances`、`invalid_vgg_instances` 和
`unresolved_geometry_images`，不得把占位项冒充真实 QR 实例。

## 7. 生成新的人工复核包

```bash
python -m barber_label_repair_zxingcpp_v2.review pack \
  --dataset "$QR_DATA" --work "$QR_WORK" --page-size 20
```

新预览只能绘制 `vgg_geometry_manifest.jsonl` 中的 VGG 变换后坐标。不得读取或绘制旧
TXT。人工检查前随机抽取不少于 30 张，确认框位置与图片一致；至少包括：

- 10 张单 QR；
- 10 张多 QR；
- 5 张此前 transform mismatch；
- 全部 invalid VGG polygon 或 source mapping failure（如数量超过 5，至少抽 5 张）。

若抽查仍发现 bbox 位置错误，停止，不填写 decision template。

## 8. 本轮停止点与报告

本轮只完成 geometry doctor、420 张 audit 和新复核包。不要执行人工决定或 finalize。

最终报告必须包含：

- branch、commit、`git status --short`；
- 16 个测试结果；
- VGG resolved/unresolved image 数、valid/invalid instance 数；
- 799/915 gold cross-validation、最大顶点误差；
- 旧 calibration 的 758、65497、1.0、740 是否成功复用；
- 420 张 ZA/ZB/ZM 图片数和实例数；
- auditable、invalid、unresolved geometry 统计；
- transform mismatch 和 VGG parse error 明细数量；
- combined proposed 图片数/实例数；
- validation report；
- 新复核包页数、路径和 30 张抽查结论；
- 原 images、旧 labels、V3 gold 未改变的确认。

任何失败都报告真实 traceback 和报告文件内容，不要临时改代码。
