#!/usr/bin/env bash
# Train the baseline pinball-QR backbone used for ordinal-CQR post-hoc processing,
# for seeds 1--4. Seed 0 is the existing reference run.
#
# Usage:
#   scripts/slurm/submit_surya_ocqr_seeds.sh
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
    --job-name="surya-ocqr-s${seed}" \
    "$TRAIN_SCRIPT" qr "$seed" \
    "experiment.ckpt_file_name=resnet18_qr")
  printf 'seed=%s job=%s method=ordinal-cqr-backbone\n' "$seed" "$job_id"
}

submit_seed 1
submit_seed 2
submit_seed 3
submit_seed 4
