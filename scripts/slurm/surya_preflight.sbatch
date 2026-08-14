#!/usr/bin/env bash
#SBATCH --job-name=surya-preflight
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Slurm stages this script under /var/spool, so resolve the helper from the
# submission repository rather than from this script's runtime location.
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/surya_common.sh"

CONFIG="${1:-configs/qr/QR_resnet18_train_surya_bench.yaml}"
python scripts/experiments/check_surya_setup.py "$CONFIG"
