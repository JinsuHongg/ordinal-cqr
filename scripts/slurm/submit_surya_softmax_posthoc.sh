#!/usr/bin/env bash
# Submit LAC, APS, and OAPS post-hoc calibration/test evaluation for the five
# Softmax classifier seeds. Each seed writes to an isolated result directory.
#
# Usage:
#   scripts/slurm/submit_surya_softmax_posthoc.sh
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
  local result_dir="$OUTPUT_ROOT/cls/seed_${seed}/posthoc/lac_aps_oaps"
  local job_id

  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint does not exist: $checkpoint" >&2
    return 2
  fi

  job_id=$(sbatch --parsable \
    --export="ALL,SURYA_CHECKPOINT=$checkpoint" \
    --job-name="surya-softmax-cal-s${seed}" \
    "$CALIBRATE_SCRIPT" cls \
    'uc.methods=[lac,aps,oaps]' \
    "uc.csv_path=$result_dir")
  printf 'seed=%s job=%s checkpoint=%s results=%s\n' \
    "$seed" "$job_id" "$(basename "$checkpoint")" "$result_dir"
}

submit_seed 0 "$OUTPUT_ROOT/cls/seed_0/training/job_4093485/checkpoints/g83dvez2_resnet18_cls_oaps_epoch=2-val_loss=0.8844.ckpt"
submit_seed 1 "$OUTPUT_ROOT/cls/seed_1/training/job_4093486/checkpoints/9onrtem6_resnet18_cls_oaps_epoch=2-val_loss=0.8976.ckpt"
submit_seed 2 "$OUTPUT_ROOT/cls/seed_2/training/job_4093487/checkpoints/rku35iry_resnet18_cls_oaps_epoch=3-val_loss=0.9007.ckpt"
submit_seed 3 "$OUTPUT_ROOT/cls/seed_3/training/job_4093488/checkpoints/x4bcb98q_resnet18_cls_oaps_epoch=4-val_loss=0.9083.ckpt"
submit_seed 4 "$OUTPUT_ROOT/cls/seed_4/training/job_4093489/checkpoints/i7fai8uy_resnet18_cls_oaps_epoch=3-val_loss=0.8856.ckpt"
