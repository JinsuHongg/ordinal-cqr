#!/usr/bin/env bash
# Submit the selected Surya QR regularization configuration for seeds 1, 2, 3, 4.
#
# Selected from the seed-0 ablation: ResNet-18 dropout=0.5, AdamW weight_decay=0.05.
# Usage:
#   scripts/slurm/submit_surya_qr_wd_0p05_seeds.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/slurm/surya_train.sbatch"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Training script not found: $TRAIN_SCRIPT" >&2
  exit 2
fi
submit_seed() {
  local seed="$1"
  local job_id
  job_id=$(sbatch --parsable \
    --job-name="surya-qr-wd005-s${seed}" \
    "$TRAIN_SCRIPT" qr "$seed" \
    "experiment.ckpt_file_name=resnet18_qr_wd_0p05" \
    "model.resnet18.p_drop=0.5" \
    "optimizer.weight_decay=0.05")
  printf 'seed=%s job=%s dropout=0.5 weight_decay=0.05\n' "$seed" "$job_id"
}
submit_seed 1
submit_seed 2
submit_seed 3
submit_seed 4
