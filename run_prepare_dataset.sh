#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/pub1/z00919662/dataset/qr_multi_240x320}"
BACKGROUND_DIR="${BACKGROUND_DIR:-/data/pub1/z00919662/dataset/coco_ADE_12cls}"

python3 "$CODE_DIR/prepare_qr_dataset.py" synthetic \
  --output "$DATA_ROOT" \
  --background-dir "$BACKGROUND_DIR" \
  --rotate-landscape-cw \
  --max-qrs-per-image "${MAX_QRS_PER_IMAGE:-5}" \
  --negative-ratio "${NEGATIVE_RATIO:-0.15}" \
  --train-count "${TRAIN_COUNT:-40000}" \
  --val-count "${VAL_COUNT:-4000}" \
  --test-count "${TEST_COUNT:-4000}"

python3 "$CODE_DIR/validate_qr_dataset.py" --data-root "$DATA_ROOT" --visualize 32

