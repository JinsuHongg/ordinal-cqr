# Running SuryaBench on a Slurm cluster

The Surya configurations are portable only after the data locations are supplied
at submission time. From the repository root, export the paths visible to the
compute nodes:

```bash
export PROJECT_DIR="$PWD"
export CONDA_ENV=ocqr
export SURYA_INDEX_DIR=/cluster/path/to/flare-index-csvs
export SURYA_ZARR_PATH=/cluster/path/to/surya-bench-224.zarr
export SURYA_STATS_PATH=/cluster/path/to/surya_channel_stats_224.yaml
export SURYA_LIMB_MASK_PATH=/cluster/path/to/limb_mask.npy
```

`SURYA_INDEX_DIR` must contain `train.csv`, `validation.csv`, `test.csv`, and
`leaky_validation.csv`. The active Surya configs require a 224×224 Zarr store
with 13 channels and a matching 224×224 limb mask. Its statistics YAML must
provide a 13-value `mean` list and a 13-value `std` list in the Zarr channel
order. The default configuration is disk-only: the mask is applied to every
normalized input channel, including AIA. All four exported paths must be
readable from a GPU node, not just the login node.

If the limb mask does not exist yet, create it first. The job selects the first
HMI-like channel from the first Zarr year group and writes a binary mask matching
the Zarr image shape. Review the printed center and radius; pass `--channel`,
`--year`, or all of `--center-x`, `--center-y`, and `--radius` to override its
automatic selection when needed:

```bash
sbatch scripts/slurm/surya_compute_limb_mask.sbatch
```

Next, if the per-channel statistics do not exist yet, set `SURYA_STATS_PATH` to
the intended output file and submit this CPU job once. It uses only retained
training timestamps, never validation, calibration, or test rows:

```bash
sbatch scripts/slurm/surya_compute_stats.sbatch
```

To evaluate option 2 (disk-masked HMI, full-frame AIA), use a **separate**
statistics file and apply the same policy to training and calibration. The
following assumes HMI channel names begin with `hmi`; confirm the names printed
by the statistics job before running it.

```bash
export SURYA_STATS_PATH=/cluster/path/to/stats_224_hmi_disk_aia_full.yaml
sbatch scripts/slurm/surya_compute_stats.sbatch --mask-mode matching --mask-channel-prefix hmi
sbatch scripts/slurm/surya_train.sbatch qr 'data.apply_limb_mask=false' 'data.limb_mask_channel_prefixes=[hmi]'
```

Use the identical two Hydra overrides when calibrating that checkpoint. If the
Zarr names do not use an `hmi` prefix, replace it with exact names via
`data.limb_mask_channels=[name1,name2]` and repeat them as
`--mask-channel name1 --mask-channel name2` for the statistics job.

Run the preflight job before using GPUs. It validates the declared SHA-256 for
all four CSVs, the retained target/label/bin contract, exact pairwise timestamp
separation, future-test chronology, Zarr layout, statistics, and mask. It also
records timestamps in different splits that are less than 24 hours apart as a
dependence diagnostic; proximity is not silently treated as direct leakage.

```bash
sbatch scripts/slurm/surya_preflight.sbatch configs/qr/QR_resnet18_train_surya_bench.yaml
```

First run a short resource and throughput smoke job. The second positional
argument is the seed; subsequent arguments are Hydra overrides. Use its
epoch runtime and the preflight's planned-step upper bound to set an appropriate
Slurm time request for the full 300-epoch job.

```bash
sbatch scripts/slurm/surya_train.sbatch qr 0 \
  trainer.max_epochs=1 trainer.limit_train_batches=20 trainer.limit_val_batches=5
```

Train one backbone at a time for each frozen seed:

```bash
sbatch scripts/slurm/surya_train.sbatch qr 0
sbatch scripts/slurm/surya_train.sbatch cls 0
sbatch scripts/slurm/surya_train.sbatch copoc 0
```

`cls` supplies the shared Softmax backbone for LAC, APS, and OAPS. `binomial`
remains available for supplementary legacy comparisons but is not canonical
COPOC. Repeat with seeds 1--4 only after the seed-0 smoke and full run pass.

Every training job writes to a collision-free path:

```text
outputs/conference_v0_3/solar_flare/<method>/seed_<seed>/training/job_<slurm_job_id>/
  checkpoints/
  resolved_config.yaml
  split_audit.json
  provenance.json
  run_status.json
```

The run record includes the resolved effective configuration and hash, Git
commit/dirty state, Python/PyTorch/Lightning/CUDA versions, Slurm identifiers,
selected validation checkpoint and loss, runtime, and split-audit hash.

Use the validation-selected checkpoint recorded as `selected_checkpoint` in the
training run's `provenance.json`. Calibration deliberately requires the complete
checkpoint path and rejects missing files before allocating a GPU:

```bash
export SURYA_CHECKPOINT="$PROJECT_DIR/outputs/conference_v0_3/solar_flare/qr/seed_0/training/job_<job-id>/checkpoints/<selected-checkpoint>.ckpt"
sbatch scripts/slurm/surya_calibrate.sbatch qr
```

Use the matching `cls` or `copoc` training path when calibrating those methods.
Do not substitute `last.ckpt` merely for convenience: the checkpoint-selection
criterion is validation loss, and the selected checkpoint path is the auditable
model-selection result.

For a train-to-calibrate dependency, capture the training job ID and use
`sbatch --dependency=afterok:<job-id> ...`. Add your cluster's account,
partition, and GPU-type directives to the top of each `.sbatch` file.

The upstream filename `leaky_validation.csv` is retained for source
compatibility. Direct timestamp non-overlap must be evidenced by each generated
`split_audit.json`; the historical filename is not itself evidence of leakage.
Because labels summarize a 24-hour forecast window, report the audit's
cross-split 24-hour proximity counts as a temporal-dependence limitation even
when exact timestamp intersections are zero.
