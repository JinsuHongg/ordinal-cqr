#!/usr/bin/env bash
# Shell helpers shared by the Surya Slurm submission scripts.

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$COMMON_DIR/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-ocqr}"

cd "$PROJECT_DIR"

if [[ -n "${CONDA_BASE:-}" ]]; then
  # Cluster-specific alternative when conda is not initialized by the shell.
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "conda is unavailable; load it first or export CONDA_BASE." >&2
  exit 2
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

: "${SURYA_INDEX_DIR:?Export SURYA_INDEX_DIR (directory with train/validation/test CSVs).}"
: "${SURYA_ZARR_PATH:?Export SURYA_ZARR_PATH (the SDO Zarr store).}"
: "${SURYA_STATS_PATH:?Export SURYA_STATS_PATH (los_mag_stat.yaml).}"
: "${SURYA_LIMB_MASK_PATH:?Export SURYA_LIMB_MASK_PATH (limb_mask.npy).}"

export SURYA_OUTPUT_ROOT="${SURYA_OUTPUT_ROOT:-$PROJECT_DIR/outputs/conference_v0_3/solar_flare}"
mkdir -p logs "$SURYA_OUTPUT_ROOT"
