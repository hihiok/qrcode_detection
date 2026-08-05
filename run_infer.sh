#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSD_ROOT="${FSD_ROOT:-/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param}"
CHECKPOINT="${CHECKPOINT:-$FSD_ROOT/models/qr_fsd_240x320_corners8/qr_fsd_best.pth}"

if [[ -z "${INPUT_PATH:-}" ]]; then
  echo "Set INPUT_PATH to a 240x320 portrait image or folder." >&2
  exit 2
fi

python3 "$CODE_DIR/infer_fsd_qr.py" \
  --fsd-repo "$FSD_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT_PATH" \
  --output "${OUTPUT_PATH:-./qr_infer_output}" \
  --input-mode y \
  --score-threshold "${SCORE_THRESHOLD:-0.5}"
