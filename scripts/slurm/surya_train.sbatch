#!/usr/bin/env bash
#SBATCH --job-name=surya-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Optional site-specific directives, e.g. --partition, --account, or --constraint,
# belong above. Do not add placeholder values: Slurm treats them as real requests.
source scripts/slurm/surya_common.sh

METHOD="${1:-qr}"
case "$METHOD" in
  qr) CONFIG_DIR=configs/qr; CONFIG_NAME=QR_resnet18_train_surya_bench ;;
  cls) CONFIG_DIR=configs/cls; CONFIG_NAME=CLS_resnet18_train_surya_bench ;;
  binomial) CONFIG_DIR=configs/cls; CONFIG_NAME=CLS_resnet18_binomial_train_surya_bench ;;
  copoc) CONFIG_DIR=configs/cls; CONFIG_NAME=CLS_resnet18_copoc_train_surya_bench ;;
  *) echo "Usage: sbatch $0 {qr|cls|binomial|copoc} [Hydra overrides...]" >&2; exit 2 ;;
esac
shift || true

python scripts/experiments/check_surya_setup.py "$CONFIG_DIR/$CONFIG_NAME.yaml"
python scripts/experiments/training.py --config-path "../../$CONFIG_DIR" --config-name "$CONFIG_NAME" "$@"
