# CodeAgent：运行已实现的 BarBeR ZXing-C++ V2

旧 V1 存在假实现、错误的 glob 切片、空白图伪结果和 payload 泄露风险。
不要继续修改或运行旧 V1。使用 GitHub 上已经实现的 V2。

## 1. 停止旧任务

先确认没有旧 V1 进程仍在写工作目录：

    pgrep -af 'recover_barber_labels_zxingcpp|barber_label_repair_zxingcpp_v1' || true

如果存在，只终止明确属于旧 V1 的 PID。不要使用宽泛的 pkill python。

## 2. 更新代码

    cd /mnt/ssd1/z00919662/qrcode_detection
    git status --short
    git fetch origin agent/barber-zxingcpp-v2
    git switch --create agent/barber-zxingcpp-v2 --track origin/agent/barber-zxingcpp-v2

如果当前仓库有未提交修改，先报告 git status，不得覆盖、stash或删除用户修改。

    git log -1 --oneline

## 3. 使用现有 Python 3.11 环境

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /mnt/ssd1/z00919662/qrcode_detection/.tools/zxingcpp_py311
    cd /mnt/ssd1/z00919662/qrcode_detection
    python -m pip install --no-cache-dir -r barber_label_repair_zxingcpp_v2/requirements.txt

如需代理或关闭 SSL 验证，严格读取并使用：

    /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md
    /mnt/ssd1/z00919662/qrcode_detection/proxy.md

不得安装或导入 OpenCV。

## 4. 静态检查和单元测试

    cd /mnt/ssd1/z00919662/qrcode_detection
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    rg -n 'placeholder|stub|dummy|result\.text|import cv2|from cv2' barber_label_repair_zxingcpp_v2
    pytest -q barber_label_repair_zxingcpp_v2/tests

预期：生产 Python 代码不含假实现、payload text访问或OpenCV；测试全部通过。

## 5. 运行 doctor

使用全新的 V2 工作目录，不覆盖 V1/V3：

    python -m barber_label_repair_zxingcpp_v2.cli doctor \
      --dataset /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated \
      --v3-work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_v3_work \
      --work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v2_work

doctor 必须确认 dataset=1219、V3 images=799、V3 instances=915、failed=420，
且纯白图在 return_errors false/true 下都返回0结果。禁止添加 allow-count-mismatch。

## 6. 运行 calibration

    python -m barber_label_repair_zxingcpp_v2.cli calibrate \
      --dataset /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated \
      --v3-work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_v3_work \
      --work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v2_work \
      2>&1 | tee /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v2_work/calibration_console.log

检查 calibration_report.json：

- passed=true
- matched_instances >= 50
- matched_observations >= 100
- modal_mapping_ratio >= 0.99
- rotation_consistent_instances >= 50

若 calibration 失败，立即停止，不运行 audit；报告真实统计和 failure_reasons，
不得降低门限或伪造报告。

## 7. calibration 通过后运行 audit

    python -m barber_label_repair_zxingcpp_v2.cli audit \
      --dataset /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated \
      --v3-work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_v3_work \
      --work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v2_work \
      2>&1 | tee /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v2_work/audit_console.log

V2采用分层重试：达到ZA即停止该实例；未达到才进入更多尺寸、quiet-zone、
预处理和binarizer。

## 8. 最终报告

读取并报告：

- calibration_report.json
- recovery_report.json
- multi_qr_completion_report.json
- transform_mismatch_report.json
- validation_report.json
- pytest结果、git status --short、git diff --stat

报告modal mapping/ratio、真实观测数、ZA/ZB/ZM图片数和实例数、
fully recovered multi-QR、combined_proposed数量、5张transform mismatch状态、
不可变哈希及人工polygon token equality。

禁止 apply、commit、push；禁止修改原始labels和V3 proposed；禁止输出payload。
