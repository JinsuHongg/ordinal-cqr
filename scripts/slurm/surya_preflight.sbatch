#!/usr/bin/env bash
#SBATCH --job-name=surya-preflight
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Submit from the repository, after exporting the four SURYA_* path variables.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/surya_common.sh"

CONFIG="${1:-configs/qr/QR_resnet18_train_surya_bench.yaml}"
python scripts/experiments/check_surya_setup.py "$CONFIG"
