#!/usr/bin/env bash
#SBATCH --job-name=surya-calibrate
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source scripts/slurm/surya_common.sh

METHOD="${1:-qr}"
case "$METHOD" in
  qr)
    : "${SURYA_CHECKPOINT:?Export SURYA_CHECKPOINT as the QR checkpoint filename in assets/checkpoints/.}"
    CONFIG=configs/qr/QR_resnet18_calibration_surya_bench.yaml
    SCRIPT=scripts/experiments/calibration.py
    OVERRIDE="check_point.qr=$SURYA_CHECKPOINT"
    ;;
  cls)
    : "${SURYA_CHECKPOINT:?Export SURYA_CHECKPOINT as the classifier checkpoint filename.}"
    CONFIG=configs/cls/CLS_resnet18_calibration_surya_bench.yaml
    SCRIPT=scripts/experiments/cls_calibration.py
    OVERRIDE="check_point.resnet18_cls=$SURYA_CHECKPOINT"
    ;;
  binomial)
    : "${SURYA_CHECKPOINT:?Export SURYA_CHECKPOINT as the binomial checkpoint filename.}"
    CONFIG=configs/cls/CLS_resnet18_binomial_calibration_surya_bench.yaml
    SCRIPT=scripts/experiments/ordinal_calibration.py
    OVERRIDE="check_point.resnet18_binomial_cls=$SURYA_CHECKPOINT"
    ;;
  copoc)
    : "${SURYA_CHECKPOINT:?Export SURYA_CHECKPOINT as the COPOC checkpoint filename.}"
    CONFIG=configs/cls/CLS_resnet18_copoc_calibration_surya_bench.yaml
    SCRIPT=scripts/experiments/ordinal_calibration.py
    OVERRIDE="check_point.resnet18_copoc_cls=$SURYA_CHECKPOINT"
    ;;
  *) echo "Usage: sbatch $0 {qr|cls|binomial|copoc} [Hydra overrides...]" >&2; exit 2 ;;
esac
shift || true

python scripts/experiments/check_surya_setup.py "$CONFIG"
CONFIG_DIR="$(dirname "$CONFIG")"
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
python "$SCRIPT" --config-path "../../$CONFIG_DIR" --config-name "$CONFIG_NAME" "$OVERRIDE" "$@"
