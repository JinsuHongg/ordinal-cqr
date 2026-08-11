#!/usr/bin/env bash
#SBATCH --job-name=surya-stats
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Statistics use only training timestamps, so a GPU is not requested.
source scripts/slurm/surya_common.sh

python scripts/experiments/compute_surya_channel_stats.py \
  --zarr "$SURYA_ZARR_PATH" \
  --train-index "$SURYA_INDEX_DIR/train.csv" \
  --limb-mask "$SURYA_LIMB_MASK_PATH" \
  --output "$SURYA_STATS_PATH" \
  "$@"
