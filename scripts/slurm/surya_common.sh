#!/usr/bin/env bash
# Shell helpers shared by the Surya Slurm submission scripts.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
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

: "${SURYA_INDEX_DIR:?Export SURYA_INDEX_DIR (directory with train/validation/test CSVs).}"
: "${SURYA_ZARR_PATH:?Export SURYA_ZARR_PATH (the SDO Zarr store).}"
: "${SURYA_STATS_PATH:?Export SURYA_STATS_PATH (los_mag_stat.yaml).}"
: "${SURYA_LIMB_MASK_PATH:?Export SURYA_LIMB_MASK_PATH (limb_mask.npy).}"

mkdir -p logs assets/checkpoints/{qr,cls,copoc} assets/uc_results
