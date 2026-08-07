#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSD_ROOT="${FSD_ROOT:-/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param}"
DEFAULT_DATA_ROOTS="/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated:/mnt/ssd1/z00919662/qrcode_detection/dataset/boofcv_qr_multi_240x320_rotation_validated:/mnt/ssd1/z00919662/qrcode_detection/dataset/mendeley_qr_multi_240x320_rotation_validated"
IFS=':' read -r -a DATA_ROOT_ARRAY <<< "${DATA_ROOTS:-${DATA_ROOT:-$DEFAULT_DATA_ROOTS}}"
DATA_ARGS=()
for dataset_root in "${DATA_ROOT_ARRAY[@]}"; do
  DATA_ARGS+=(--data-root "$dataset_root")
done
OUTPUT_DIR="${OUTPUT_DIR:-$FSD_ROOT/models/qr_fsd_multi_yuv}"

WEIGHT_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  WEIGHT_ARGS=(--resume "$RESUME_CHECKPOINT")
elif [[ -n "${FD_CHECKPOINT:-}" ]]; then
  WEIGHT_ARGS=(--pretrained-fd "$FD_CHECKPOINT")
else
  echo "Set RESUME_CHECKPOINT (old QR corners model) or FD_CHECKPOINT (bbox FSD)." >&2
  exit 2
fi

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_LIST" python3 -u "$CODE_DIR/train_fsd_qr.py" \
  --fsd-repo "$FSD_ROOT" \
  "${DATA_ARGS[@]}" \
  --checkpoint-dir "$OUTPUT_DIR" \
  "${WEIGHT_ARGS[@]}" \
  --input-mode yuv \
  --input-size-key 240 \
  --batch-size "${BATCH_SIZE:-64}" \
  --num-workers "${NUM_WORKERS:-16}" \
  --epochs "${EPOCHS:-200}" \
  --gpus "$GPU_LIST" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
