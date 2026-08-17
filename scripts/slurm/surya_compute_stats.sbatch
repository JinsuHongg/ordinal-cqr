#!/bin/bash
# GSU-specific account and partition are supplied at submission time, e.g.:
#   sbatch --account=RS00000 --partition=qBF scripts/slurm/surya_compute_stats.sbatch
# Optional mail settings can be supplied the same way:
#   sbatch --mail-type=END,FAIL --mail-user=you@gsu.edu ...
# Do not use shell variables or `{placeholders}` in #SBATCH lines: Slurm reads
# these directives before Bash expands variables.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=surya-stats
#SBATCH --output=output_%x_%j.txt
#SBATCH --error=error_%x_%j.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Statistics use only training timestamps, so a GPU is not requested.
# `PROJECT_DIR` should be exported in the submission environment. Falling back
# to Slurm's submit directory also supports `sbatch` from the repository root.
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/surya_common.sh"

python scripts/experiments/compute_surya_channel_stats.py \
  --zarr "$SURYA_ZARR_PATH" \
  --train-index "$SURYA_INDEX_DIR/train.csv" \
  --limb-mask "$SURYA_LIMB_MASK_PATH" \
  --output "$SURYA_STATS_PATH" \
  --consolidate-metadata \
  "$@"
