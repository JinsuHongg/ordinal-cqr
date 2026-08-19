#!/usr/bin/env bash
# Submit three OCQR post-processing ablations for each of five QR checkpoints.
# The canonical fallback+hull result already exists; this launcher runs only:
#   no_fallback, no_hull, and raw (neither fallback nor hull).
#
# Usage:
#   scripts/slurm/submit_surya_ocqr_postprocessing_ablations.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CALIBRATE_SCRIPT="$PROJECT_DIR/scripts/slurm/surya_calibrate.sbatch"
OUTPUT_ROOT="${SURYA_OUTPUT_ROOT:-$PROJECT_DIR/outputs/conference_v0_3/solar_flare}"

if [[ ! -f "$CALIBRATE_SCRIPT" ]]; then
  echo "Calibration script not found: $CALIBRATE_SCRIPT" >&2
  exit 2
fi

submit_variant() {
  local seed="$1"
  local checkpoint="$2"
  local variant="$3"
  local fallback="$4"
  local hull="$5"
  local result_dir="$OUTPUT_ROOT/qr/seed_${seed}/posthoc/ocqr_${variant}"
  local job_id

  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint does not exist: $checkpoint" >&2
    return 2
  fi

  job_id=$(sbatch --parsable \
    --export="ALL,SURYA_CHECKPOINT=$checkpoint" \
    --job-name="surya-ocqr-${variant}-s${seed}" \
    "$CALIBRATE_SCRIPT" qr \
    'uc.methods=[ordinal_cqr]' \
    "uc.ordinal_cqr.apply_empty_set_fallback=$fallback" \
    "uc.ordinal_cqr.enforce_ordinal_hull=$hull" \
    "uc.csv_path=$result_dir")
  printf 'seed=%s variant=%s job=%s checkpoint=%s\n' \
    "$seed" "$variant" "$job_id" "$(basename "$checkpoint")"
}

submit_seed() {
  local seed="$1"
  local checkpoint="$2"
  submit_variant "$seed" "$checkpoint" no_fallback false true
  submit_variant "$seed" "$checkpoint" no_hull true false
  submit_variant "$seed" "$checkpoint" raw false false
}

submit_seed 0 "$OUTPUT_ROOT/qr/seed_0/training/job_4057072/checkpoints/bu2z1vlt_resnet18_qr_q89_epoch=5-val_loss=0.1144.ckpt"
submit_seed 1 "$OUTPUT_ROOT/qr/seed_1/training/job_4093739/checkpoints/4ilpfvmp_resnet18_qr_q89_epoch=6-val_loss=0.1140.ckpt"
submit_seed 2 "$OUTPUT_ROOT/qr/seed_2/training/job_4093740/checkpoints/zisv7wkz_resnet18_qr_q89_epoch=4-val_loss=0.1139.ckpt"
submit_seed 3 "$OUTPUT_ROOT/qr/seed_3/training/job_4093741/checkpoints/vtf6guyx_resnet18_qr_q89_epoch=8-val_loss=0.1053.ckpt"
submit_seed 4 "$OUTPUT_ROOT/qr/seed_4/training/job_4093742/checkpoints/tf44083n_resnet18_qr_q89_epoch=6-val_loss=0.1119.ckpt"
