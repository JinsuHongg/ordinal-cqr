#!/usr/bin/env bash
#SBATCH --job-name=surya-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Optional site-specific directives, e.g. --partition, --account, or --constraint,
# belong above. Do not add placeholder values: Slurm treats them as real requests.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/surya_common.sh"

METHOD="${1:-qr}"
case "$METHOD" in
  qr) CONFIG_DIR=configs/qr; CONFIG_NAME=QR_resnet18_train_surya_bench ;;
  cls) CONFIG_DIR=configs/cls; CONFIG_NAME=CLS_resnet18_train_surya_bench ;;
  binomial) CONFIG_DIR=configs/cls; CONFIG_NAME=CLS_resnet18_binomial_train_surya_bench ;;
  copoc) CONFIG_DIR=configs/cls; CONFIG_NAME=CLS_resnet18_copoc_train_surya_bench ;;
  *) echo "Usage: sbatch $0 {qr|cls|binomial|copoc} [Hydra overrides...]" >&2; exit 2 ;;
esac
shift || true

SEED="${SURYA_SEED:-0}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  SEED="$1"
  shift
fi

JOB_TOKEN="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$SURYA_OUTPUT_ROOT/$METHOD/seed_$SEED/training/job_$JOB_TOKEN"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/checkpoints"
export SURYA_RUN_DIR="$RUN_DIR"
export SURYA_METHOD="$METHOD"

GENERATED_OVERRIDES=(
  "seed=$SEED"
  "model.save_ckpt_path=$RUN_DIR/checkpoints"
  "trainer.deterministic=true"
)

printf 'Surya training run: method=%s seed=%s job=%s output=%s\n' \
  "$METHOD" "$SEED" "$JOB_TOKEN" "$RUN_DIR"

python scripts/experiments/check_surya_setup.py \
  --audit-output "$RUN_DIR/split_audit.json" \
  "$CONFIG_DIR/$CONFIG_NAME.yaml" "${GENERATED_OVERRIDES[@]}" "$@"
python scripts/experiments/training.py \
  --config-path "../../$CONFIG_DIR" --config-name "$CONFIG_NAME" \
  "${GENERATED_OVERRIDES[@]}" "$@"
