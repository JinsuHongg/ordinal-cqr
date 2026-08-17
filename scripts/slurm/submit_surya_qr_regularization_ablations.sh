#!/usr/bin/env bash
# Submit the seed-controlled QR regularization ablations for SuryaBench.
#
# Usage:
#   scripts/slurm/submit_surya_qr_regularization_ablations.sh [seed]
#
# Required environment: the same SURYA_* variables used by surya_train.sbatch.
# The launcher only submits jobs; it does not modify data, statistics, or
# checkpoints. Each training job creates its own output directory.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/slurm/surya_train.sbatch"
SEED="${1:-0}"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [non-negative seed]" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Training script not found: $TRAIN_SCRIPT" >&2
  exit 2
fi
submit_ablation() {
  local label="$1"
  local dropout="$2"
  local weight_decay="$3"
  local job_id
  job_id=$(sbatch --parsable \
    --job-name="surya-qr-$label" \
    "$TRAIN_SCRIPT" qr "$SEED" \
    "experiment.ckpt_file_name=resnet18_qr_$label" \
    "model.resnet18.p_drop=$dropout" \
    "optimizer.weight_decay=$weight_decay")
  printf '%-22s job=%s seed=%s dropout=%s weight_decay=%s\n' \
    "$label" "$job_id" "$SEED" "$dropout" "$weight_decay"
}
# One-factor ablations isolate the regularizer; the third run tests whether
# their combination improves validation pinball loss beyond either alone.
submit_ablation "wd_0p05" 0.5 0.05
submit_ablation "dropout_0p70" 0.7 0.01
submit_ablation "dropout_0p70_wd_0p05" 0.7 0.05
