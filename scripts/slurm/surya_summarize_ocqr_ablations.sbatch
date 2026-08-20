#!/usr/bin/env bash
#SBATCH --job-name=surya-ablation-summary
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
python scripts/experiments/summarize_surya_ocqr_ablations.py \
  --root outputs/conference_v0_3/solar_flare/qr \
  --output-dir docs/results
