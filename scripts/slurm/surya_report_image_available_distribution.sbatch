#!/usr/bin/env bash
#SBATCH --job-name=surya-image-dist
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/surya_common.sh"

OUTPUT="$SURYA_OUTPUT_ROOT/data_audits/image_available_distribution_job_${SLURM_JOB_ID}.json"
python scripts/experiments/report_surya_image_available_distribution.py \
  --index-dir "$SURYA_INDEX_DIR" \
  --zarr-path "$SURYA_ZARR_PATH" \
  --output "$OUTPUT"
