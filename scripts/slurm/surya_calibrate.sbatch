#!/usr/bin/env bash
#SBATCH --job-name=surya-calibrate
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Slurm stages this script under /var/spool, so resolve the helper from the
# submission repository rather than from this script's runtime location.
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/surya_common.sh"

METHOD="${1:-qr}"
: "${SURYA_CHECKPOINT:?Export SURYA_CHECKPOINT as the trained checkpoint path.}"
if [[ ! -f "$SURYA_CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $SURYA_CHECKPOINT" >&2
  exit 2
fi
CHECKPOINT_PATH="$(realpath "$SURYA_CHECKPOINT")"
CHECKPOINT_DIR="$(dirname "$CHECKPOINT_PATH")"
CHECKPOINT_NAME="$(basename "$CHECKPOINT_PATH")"

case "$METHOD" in
  qr)
    CONFIG=configs/qr/QR_resnet18_calibration_surya_bench.yaml
    SCRIPT=scripts/experiments/calibration.py
    OVERRIDE="check_point.qr='${CHECKPOINT_NAME}'"
    ;;
  cls)
    CONFIG=configs/cls/CLS_resnet18_calibration_surya_bench.yaml
    SCRIPT=scripts/experiments/cls_calibration.py
    OVERRIDE="check_point.resnet18_cls='${CHECKPOINT_NAME}'"
    ;;
  binomial)
    CONFIG=configs/cls/CLS_resnet18_binomial_calibration_surya_bench.yaml
    SCRIPT=scripts/experiments/ordinal_calibration.py
    OVERRIDE="check_point.resnet18_binomial_cls='${CHECKPOINT_NAME}'"
    ;;
  copoc)
    CONFIG=configs/cls/CLS_resnet18_copoc_calibration_surya_bench.yaml
    SCRIPT=scripts/experiments/ordinal_calibration.py
    OVERRIDE="check_point.resnet18_copoc_cls='${CHECKPOINT_NAME}'"
    ;;
  *) echo "Usage: sbatch $0 {qr|cls|binomial|copoc} [Hydra overrides...]" >&2; exit 2 ;;
esac
shift || true

BASE_OVERRIDE="check_point.base=$CHECKPOINT_DIR"
python scripts/experiments/check_surya_setup.py \
  "$CONFIG" "$BASE_OVERRIDE" "$OVERRIDE" "$@"
CONFIG_DIR="$(dirname "$CONFIG")"
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
python "$SCRIPT" --config-path "../../$CONFIG_DIR" --config-name "$CONFIG_NAME" \
  "$BASE_OVERRIDE" "$OVERRIDE" "$@"
