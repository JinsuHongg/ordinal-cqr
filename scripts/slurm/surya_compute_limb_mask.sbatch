#!/bin/bash
# GSU-specific account and partition are supplied at submission time, e.g.:
#   sbatch --account=RS00000 --partition=qBF scripts/slurm/surya_compute_limb_mask.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=surya-limb-mask
#SBATCH --output=output_%x_%j.txt
#SBATCH --error=error_%x_%j.txt
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV="${CONDA_ENV:-ocqr}"
cd "$PROJECT_DIR"

if [[ -n "${CONDA_BASE:-}" ]]; then
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

: "${SURYA_ZARR_PATH:?Export SURYA_ZARR_PATH (the SDO Zarr store).}"
: "${SURYA_LIMB_MASK_PATH:?Export SURYA_LIMB_MASK_PATH (output limb_mask.npy).}"

python scripts/experiments/compute_surya_limb_mask.py \
  --zarr "$SURYA_ZARR_PATH" \
  --output "$SURYA_LIMB_MASK_PATH" \
  "$@"
