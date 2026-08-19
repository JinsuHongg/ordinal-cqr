#!/usr/bin/env bash
# Submit the exploratory OCQR nonnegative-correction variant for seeds 0--4.
# It retains canonical fallback and hull behavior but uses max(q_k, 0) during
# candidate-set inversion. The original calibrated q_k remains in metadata.
#
# Usage:
#   scripts/slurm/submit_surya_ocqr_nonnegative_correction.sh
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
  local result_dir="$OUTPUT_ROOT/qr/seed_${seed}/posthoc/ocqr_nonnegative_correction"
  local job_id

  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint does not exist: $checkpoint" >&2
    return 2
  fi

  job_id=$(sbatch --parsable \
    --export="ALL,SURYA_CHECKPOINT=$checkpoint" \
    --job-name="surya-ocqr-nonneg-s${seed}" \
    "$CALIBRATE_SCRIPT" qr \
    'uc.methods=[ordinal_cqr]' \
    'uc.ordinal_cqr.apply_empty_set_fallback=true' \
    'uc.ordinal_cqr.enforce_ordinal_hull=true' \
    'uc.ordinal_cqr.clip_corrections_nonnegative=true' \
    "uc.csv_path=$result_dir")
  printf 'seed=%s job=%s checkpoint=%s results=%s\n' \
    "$seed" "$job_id" "$(basename "$checkpoint")" "$result_dir"
}

submit_seed 0 "$OUTPUT_ROOT/qr/seed_0/training/job_4057072/checkpoints/bu2z1vlt_resnet18_qr_q89_epoch=5-val_loss=0.1144.ckpt"
submit_seed 1 "$OUTPUT_ROOT/qr/seed_1/training/job_4093739/checkpoints/4ilpfvmp_resnet18_qr_q89_epoch=6-val_loss=0.1140.ckpt"
submit_seed 2 "$OUTPUT_ROOT/qr/seed_2/training/job_4093740/checkpoints/zisv7wkz_resnet18_qr_q89_epoch=4-val_loss=0.1139.ckpt"
submit_seed 3 "$OUTPUT_ROOT/qr/seed_3/training/job_4093741/checkpoints/vtf6guyx_resnet18_qr_q89_epoch=8-val_loss=0.1053.ckpt"
submit_seed 4 "$OUTPUT_ROOT/qr/seed_4/training/job_4093742/checkpoints/tf44083n_resnet18_qr_q89_epoch=6-val_loss=0.1119.ckpt"
