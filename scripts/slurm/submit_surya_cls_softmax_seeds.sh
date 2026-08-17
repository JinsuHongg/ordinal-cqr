#!/usr/bin/env bash
# Train the standard Softmax backbone for five-seed Surya classification baselines.
#
# OAPS is applied after training by calibrating this cls checkpoint with
# scripts/slurm/surya_calibrate.sbatch cls. This backbone is also valid for
# LAC and APS comparisons.
#
# Usage:
#   scripts/slurm/submit_surya_cls_softmax_seeds.sh
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
    --job-name="surya-cls-softmax-s${seed}" \
    "$TRAIN_SCRIPT" cls "$seed" \
    "experiment.ckpt_file_name=resnet18_cls_softmax")
  printf 'seed=%s job=%s method=softmax_cls downstream=lac,aps,oaps\n' "$seed" "$job_id"
}
submit_seed 0
submit_seed 1
submit_seed 2
submit_seed 3
submit_seed 4
