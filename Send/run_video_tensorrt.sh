#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_MODEL="${1:-$SCRIPT_DIR/hybridnets_road_segmentation.onnx}"
INPUT_VIDEO="${2:-$SCRIPT_DIR/input.mp4}"
OUTPUT_VIDEO="${3:-$SCRIPT_DIR/output_stanley_trt.mp4}"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/home/aman/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/home/aman/miniconda3/etc/profile.d/conda.sh"
else
  echo "Unable to find conda.sh. Activate hybridnets-local manually and rerun." >&2
  exit 1
fi

conda activate hybridnets-local
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-hybridnets-send"

python "$SCRIPT_DIR/InferRoadVideoTensorRT.py" \
  --onnx "$ONNX_MODEL" \
  --video "$INPUT_VIDEO" \
  --output "$OUTPUT_VIDEO" \
  "${@:4}"
