#!/usr/bin/env bash
# Submit COPOC post-hoc calibration/test evaluation for five trained seeds.
# Each seed writes to an isolated result directory.
#
# Usage:
#   scripts/slurm/submit_surya_copoc_posthoc.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CALIBRATE_SCRIPT="$PROJECT_DIR/scripts/slurm/surya_calibrate.sbatch"
OUTPUT_ROOT="${SURYA_OUTPUT_ROOT:-$PROJECT_DIR/outputs/conference_v0_3/solar_flare}"

if [[ ! -f "$CALIBRATE_SCRIPT" ]]; then
  echo "Calibration script not found: $CALIBRATE_SCRIPT" >&2
  exit 2
fi

submit_seed() {
  local seed="$1"
  local checkpoint="$2"
  local result_dir="$OUTPUT_ROOT/copoc/seed_${seed}/posthoc/copoc"
  local job_id

  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint does not exist: $checkpoint" >&2
    return 2
  fi

  job_id=$(sbatch --parsable \
    --export="ALL,SURYA_CHECKPOINT=$checkpoint" \
    --job-name="surya-copoc-cal-s${seed}" \
    "$CALIBRATE_SCRIPT" copoc \
    'uc.methods=[copoc]' \
    "uc.csv_path=$result_dir")
  printf 'seed=%s job=%s checkpoint=%s results=%s\n' \
    "$seed" "$job_id" "$(basename "$checkpoint")" "$result_dir"
}

submit_seed 0 "$OUTPUT_ROOT/copoc/seed_0/training/job_4093490/checkpoints/ueaaoty9_resnet18_copoc_surya_bench_epoch=4-val_loss=0.9471.ckpt"
submit_seed 1 "$OUTPUT_ROOT/copoc/seed_1/training/job_4093491/checkpoints/ib4viazj_resnet18_copoc_surya_bench_epoch=4-val_loss=0.9435.ckpt"
submit_seed 2 "$OUTPUT_ROOT/copoc/seed_2/training/job_4093492/checkpoints/q066zb2q_resnet18_copoc_surya_bench_epoch=4-val_loss=0.9310.ckpt"
submit_seed 3 "$OUTPUT_ROOT/copoc/seed_3/training/job_4093493/checkpoints/up21ivfg_resnet18_copoc_surya_bench_epoch=4-val_loss=0.9377.ckpt"
submit_seed 4 "$OUTPUT_ROOT/copoc/seed_4/training/job_4093494/checkpoints/ijks89m1_resnet18_copoc_surya_bench_epoch=3-val_loss=0.9422.ckpt"
