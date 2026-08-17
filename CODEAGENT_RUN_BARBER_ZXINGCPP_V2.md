# CodeAgent 完整执行：BarBeR 剩余 420 张审计与人工收尾

> **已废弃：不要执行本文档。** V2 错误地把处理后数据集旧 TXT 当作 polygon
> 几何来源。请改为执行 `CODEAGENT_RUN_BARBER_ZXINGCPP_V3_GEOMETRY.md`。

## 目标与完成定义

继续使用已通过校准的 ZXing-C++ V2，完成剩余 420 张图片的真实审计，生成
ZA/ZB/ZM 结果和人工复核包。人工逐张确认 ZB/ZM 后，创建一个新的完整 BarBeR
数据集；不得覆盖原数据集。

完整数据集必须满足：

- 1219 张图片全部有最终标签；
- 一张图内所有 QR 实例都已确认；
- P0/P1/P2/P3 是二维码自身语义方向；
- 自动恢复标签的坐标 token 只来自原 BarBeR 人工 polygon；
- 退化 polygon、parse error、transform mismatch 必须由人工提供修正标签；
- 原 images、原 labels、V3 799 张/915 实例 gold 全部哈希不变。

本任务分两个阶段。阶段 A 可立即执行到结束。阶段 B 必须等待人工填写复核决定，
禁止 CodeAgent 冒充人工确认或把 `pending` 自动改成通过。

## 固定路径

    export QR_REPO=/mnt/ssd1/z00919662/qrcode_detection
    export QR_RUNNER=/mnt/ssd1/z00919662/qrcode_detection_zxingcpp_v2_runner
    export QR_DATA=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated
    export QR_V3=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_v3_work
    export QR_WORK=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v2_work
    export QR_FINAL=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_complete_reviewed

代码分支：`agent/barber-zxingcpp-v2`。

## 全程禁止

- 不要运行旧 V1。
- 不要修改工具代码；若最新版仍报错，保留现场并报告，不要临时降低规则。
- 不要重新运行已经通过的 calibration，除非报告或 mapping 缺失/不通过。
- 不要使用 `--allow-count-mismatch`。
- 不要修改或覆盖 `$QR_DATA`、`$QR_V3`。
- 不要把 ZXing position 坐标写入标签。
- 不要降低 calibration、ZA、ZB、几何门限。
- 不要输出二维码 payload；只能保留 SHA-256。
- 不要 commit、push、merge PR。
- 不要在有人工作时删除或覆盖 `$QR_FINAL`。

# 阶段 A：更新代码、续跑 420 张 audit、生成复核包

## A1. 停止旧 V1，更新独立 runner

先检查：

    pgrep -af 'recover_barber_labels_zxingcpp|barber_label_repair_zxingcpp_v1' || true

如果有进程，只终止命令行明确包含旧 V1 的 PID。禁止使用宽泛的
`pkill python`。

如果 runner 不存在：

    if [ ! -d "$QR_RUNNER/.git" ]; then
      test ! -e "$QR_RUNNER" || {
        echo "ERROR: runner path exists but is not git repo: $QR_RUNNER"
        exit 2
      }
      git clone --single-branch --branch agent/barber-zxingcpp-v2 \
        https://github.com/hihiok/qrcode_detection.git "$QR_RUNNER"
    fi

更新已有 runner。上次运行产生的两个 `__pycache__` 可以删除；其他本地改动一律
不得覆盖、stash或删除：

    cd "$QR_RUNNER"
    rm -rf \
      "$QR_RUNNER/barber_label_repair_zxingcpp_v2/__pycache__" \
      "$QR_RUNNER/barber_label_repair_zxingcpp_v2/tests/__pycache__"
    git status --short
    test -z "$(git status --porcelain)" || {
      echo "ERROR: runner contains non-cache local changes; stop and report"
      exit 2
    }
    git fetch origin agent/barber-zxingcpp-v2
    git switch agent/barber-zxingcpp-v2
    git merge --ff-only origin/agent/barber-zxingcpp-v2
    git log -1 --oneline
    git status --short

确认最新提交包含以下内容：

- `quad_geometry_issue`；
- `reset_audit_outputs`；
- `test_audit_isolates_degenerate_instance_and_removes_stale_output`；
- `barber_label_repair_zxingcpp_v2/review.py`；
- `test_review_pack_renders_and_writes_pending_template`。

## A2. 环境与静态安全检查

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /mnt/ssd1/z00919662/qrcode_detection/.tools/zxingcpp_py311
    cd "$QR_RUNNER"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python -m pip install --no-cache-dir \
      -r barber_label_repair_zxingcpp_v2/requirements.txt
    python - <<'PY'
    import sys, zxingcpp
    print("python:", sys.version)
    print("zxingcpp import: OK")
    PY

如安装需要代理或关闭 SSL 验证，只能读取并使用：

    $QR_REPO/CODEAGENT_DISABLE_SSL.md
    $QR_REPO/proxy.md

生产代码安全检查：

    if rg -n 'result\.text|import cv2|from cv2|NotImplemented' \
      barber_label_repair_zxingcpp_v2 --glob '*.py' \
      --glob '!tests/**'; then
      echo "ERROR: forbidden production pattern found"
      exit 2
    fi

## A3. 运行完整测试

    python -m pytest -q barber_label_repair_zxingcpp_v2/tests

预期至少 15 个测试通过，包括：

- 真实 zxing-cpp 正样本和空白图负样本；
- 四旋转逆映射；
- 最终坐标 token 原样复用；
- 单图端到端 ZA；
- 同图一个正常实例、一个退化实例时不崩溃；
- audit 重跑会清理旧半成品；
- 人工复核包可以真实渲染。

任一测试失败，停止，不运行 audit。

## A4. Doctor 与复用已通过的 calibration

    mkdir -p "$QR_WORK"
    python -m barber_label_repair_zxingcpp_v2.cli doctor \
      --dataset "$QR_DATA" --v3-work "$QR_V3" --work "$QR_WORK"

Doctor 必须确认：1219 总图、799 V3 gold 图、915 gold 实例、420 failed；纯白图
不能产生 ZXing 结果；首次建立的 immutable baseline 只能比较，不能重写。

检查并复用上次 calibration：

    python - <<'PY'
    import json, os
    from pathlib import Path
    work = Path(os.environ["QR_WORK"])
    report = json.loads((work / "calibration_report.json").read_text())
    mapping = json.loads((work / "calibration_mapping.json").read_text())
    assert report["passed"] is True, report
    assert report["matched_instances"] >= 50, report
    assert report["matched_observations"] >= 100, report
    assert report["modal_mapping_ratio"] >= 0.99, report
    assert report["rotation_consistent_instances"] >= 50, report
    assert sorted(mapping["field_to_semantic"]) == [0, 1, 2, 3], mapping
    print(json.dumps({
        "matched_instances": report["matched_instances"],
        "matched_observations": report["matched_observations"],
        "modal_mapping": report["modal_mapping"],
        "modal_mapping_ratio": report["modal_mapping_ratio"],
        "rotation_consistent_instances": report["rotation_consistent_instances"],
        "allow_error_results": report["allow_error_results"],
    }, indent=2))
    PY

预期能读到上次真实结果：matched instances 758、matched observations 65497、
modal ratio 1.0、rotation-consistent instances 740。若报告不存在或校验失败，停止并
报告；不要自行重跑或降低门限。

## A5. 重新运行完整 420 张 audit

新版 audit 会先清理它自己上次产生的半成品目录，但保留 calibration、mapping、
immutable baseline。退化实例记为 ZM，同图其他实例继续；单个 crop/warp 失败也不会
终止整批。

    set -o pipefail
    python -m barber_label_repair_zxingcpp_v2.cli audit \
      --dataset "$QR_DATA" --v3-work "$QR_V3" --work "$QR_WORK" \
      2>&1 | tee "$QR_WORK/audit_console.log"
    audit_rc=${PIPESTATUS[0]}
    test "$audit_rc" -eq 0 || {
      echo "ERROR: audit failed rc=$audit_rc; preserve outputs and report traceback"
      exit "$audit_rc"
    }

不要并行启动第二个 audit。

## A6. 强制验证 audit 产物

    python - <<'PY'
    import json, os
    from pathlib import Path
    work = Path(os.environ["QR_WORK"])
    recovery = json.loads((work / "recovery_report.json").read_text())
    validation = json.loads((work / "validation_report.json").read_text())
    assert recovery["input_failed_images"] == 420, recovery
    assert recovery["ZA_images"] + recovery["ZB_images"] + recovery["ZM_images"] == 420
    assert recovery["ZA_instances"] + recovery["ZB_instances"] + recovery["ZM_instances"] == recovery["input_failed_instances"]
    assert recovery["combined_proposed_images"] == 799 + recovery["ZA_images"]
    assert validation["passed"] is True, validation
    assert validation["original_images_labels_v3_unchanged"] is True
    assert validation["polygon_vertex_token_equality"] is True
    required = [
        "failed_420_manifest.jsonl", "calibration_report.json",
        "calibration_mapping.json", "calibration_evidence.jsonl",
        "recovery_report.json", "orientation_evidence.jsonl",
        "recovered_predictions.jsonl", "recovery_failures.jsonl",
        "transform_mismatch_report.json", "parse_error_report.json",
        "multi_qr_completion_report.json", "validation_report.json",
    ]
    missing = [name for name in required if not (work / name).is_file()]
    assert not missing, missing
    print(json.dumps(recovery, indent=2, sort_keys=True))
    PY

额外统计 `invalid_manual_quad`、parse error、transform mismatch：

    python - <<'PY'
    import json, os
    from collections import Counter
    from pathlib import Path
    path = Path(os.environ["QR_WORK"]) / "recovery_failures.jsonl"
    reasons = Counter()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for inst in row.get("instances", []):
            reasons[inst.get("facts", {}).get("reason", "evidence_not_ZA")] += 1
    print(json.dumps(reasons, indent=2, sort_keys=True))
    PY

## A7. 生成人工复核包

    python -m barber_label_repair_zxingcpp_v2.review pack \
      --dataset "$QR_DATA" --work "$QR_WORK" --page-size 20

输出：

    $QR_WORK/manual_review_pack/review_pack_report.json
    $QR_WORK/manual_review_pack/previews/zb_page_*.jpg
    $QR_WORK/manual_review_pack/previews/zm_page_*.jpg
    $QR_WORK/manual_review_pack/manual_review_decisions_TEMPLATE.jsonl

预览中：

- `P0/Vn` 表示建议把原人工 polygon 的第 n 个顶点作为 P0；
- `V0..V3` 是原标签四个顶点索引；
- 每张图标题含 split/image_id；
- 多二维码图会同时显示所有实例；
- ZB/ZM 都必须人工逐图检查，不能仅凭票数自动通过。

阶段 A 到此停止，并报告真实 ZA/ZB/ZM 数量以及复核包路径。不要替人工填写决定。

# 阶段 B：人工决定填写完成后，创建完整数据集

只有用户明确告知人工复核已经完成后才执行。

## B1. 决定文件格式

先复制模板，保留原模板：

    cp "$QR_WORK/manual_review_pack/manual_review_decisions_TEMPLATE.jsonl" \
       "$QR_WORK/manual_review_pack/manual_review_decisions.jsonl"

人工逐行填写 `reviewer`，并为每张图片选择一种 action：

1. `approve_suggestion`
   - 人工确认预览中的全部实例方向建议正确；
   - 保留 `orders`；每个 order 必须是 `[0,1,2,3]` 的排列。
2. `manual_order`
   - 人工修改每个实例的 `orders`；
   - `orders[实例号] = [P0对应V索引, P1对应V索引, P2对应V索引, P3对应V索引]`。
3. `corrected_label`
   - 用于退化 polygon、parse error、5 张 transform mismatch 或原 polygon 本身错误；
   - 在以下路径放入人工重标后的完整标签：
     `$QR_WORK/manual_corrected_labels/<split>/labels/<stem>.txt`；
   - 标签每行必须为：`0 P0x P0y P1x P1y P2x P2y P3x P3y`；
   - 一图多 QR 时必须包含全部实例，每个实例一行。

以下情况严禁 `approve_suggestion`：

- `invalid_manual_quad`；
- 原标签 parse error；
- 5 张 transform mismatch；
- 实例没有 candidate order。

这些必须人工重标或填写可靠的 `manual_order`；若 polygon 坐标错误，则只能使用
`corrected_label`。

## B2. Finalize 到全新目录

先确认目标不存在；不要删除已有目标：

    test ! -e "$QR_FINAL" || {
      echo "ERROR: final output already exists; do not overwrite"
      exit 2
    }

执行：

    python -m barber_label_repair_zxingcpp_v2.review finalize \
      --dataset "$QR_DATA" --v3-work "$QR_V3" --work "$QR_WORK" \
      --decisions "$QR_WORK/manual_review_pack/manual_review_decisions.jsonl" \
      --output "$QR_FINAL"

只要仍有一个图片是 pending、决定无 reviewer、order 不是四点排列、退化 polygon
没有 corrected label，finalize 必须非零退出，并写：

    $QR_WORK/manual_review_pack/finalization_pending.json

补完这些项目后，确认 `$QR_FINAL` 仍不存在，再重试。禁止跳过 unresolved 图片。

## B3. 完整数据集最终验收

    python - <<'PY'
    import json, os
    from pathlib import Path
    root = Path(os.environ["QR_FINAL"])
    report = json.loads((root / "finalization_report.json").read_text())
    assert report["passed"] is True, report
    assert report["images"] == 1219, report
    images = []
    labels = []
    for split in ("train", "val", "test"):
        images += [p for p in (root / split / "images").iterdir() if p.is_file()]
        labels += list((root / split / "labels").glob("*.txt"))
    assert len(images) == 1219, len(images)
    assert len(labels) == 1219, len(labels)
    manifest = (root / "finalization_manifest.jsonl").read_text().splitlines()
    assert len(manifest) == 1219, len(manifest)
    assert report["original_inputs_unchanged"] is True
    print(json.dumps(report, indent=2, sort_keys=True))
    PY

再次运行 doctor，验证原输入哈希仍未变化：

    python -m barber_label_repair_zxingcpp_v2.cli doctor \
      --dataset "$QR_DATA" --v3-work "$QR_V3" --work "$QR_WORK"

## 最终报告格式

阶段 A 报告：

- Git commit；测试通过数；
- calibration 关键统计和是否复用；
- 420 张的 ZA/ZB/ZM 图片数、实例数；
- fully recovered multi-QR 数；
- combined_proposed 图片/实例数；
- invalid polygon、parse error、5 张 transform mismatch 数量；
- validation_report；
- 人工复核包页数和决定文件路径；
- `git status --short`、`git diff --stat`。

阶段 B 报告：

- 人工决定总数以及三种 action 数量；
- 最终图片数、实例数、各 split 数量；
- V3_gold / ZXing_ZA / human_reviewed 来源图片数；
- finalization_report 和 manifest 行数；
- 原输入哈希不变；
- 完整数据集绝对路径；
- `git status --short`、`git diff --stat`。

不要报告二维码 payload，不要修改代码，不要 commit/push。
