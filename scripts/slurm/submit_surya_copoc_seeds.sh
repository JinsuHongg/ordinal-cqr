#!/usr/bin/env bash
# Train the canonical COPOC Eq. (5) backbone for five Surya seeds.
#
# Usage:
#   scripts/slurm/submit_surya_copoc_seeds.sh
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
    --job-name="surya-copoc-s${seed}" \
    "$TRAIN_SCRIPT" copoc "$seed" \
    "experiment.ckpt_file_name=resnet18_copoc_surya_bench")
  printf 'seed=%s job=%s method=copoc\n' "$seed" "$job_id"
}
submit_seed 0
submit_seed 1
submit_seed 2
submit_seed 3
submit_seed 4
