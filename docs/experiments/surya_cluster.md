# Running SuryaBench on a Slurm cluster

The Surya configurations are portable only after the data locations are supplied
at submission time. From the repository root, export the paths visible to the
compute nodes:

```bash
export PROJECT_DIR="$PWD"
export CONDA_ENV=ocqr
export SURYA_INDEX_DIR=/cluster/path/to/flare-index-csvs
export SURYA_ZARR_PATH=/cluster/path/to/sdo_512.zarr
export SURYA_STATS_PATH=/cluster/path/to/los_mag_stat.yaml
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

Run the preflight job before using GPUs:

```bash
sbatch scripts/slurm/surya_preflight.sbatch configs/qr/QR_resnet18_train_surya_bench.yaml
```

Train one backbone at a time:

```bash
sbatch scripts/slurm/surya_train.sbatch qr
sbatch scripts/slurm/surya_train.sbatch cls
sbatch scripts/slurm/surya_train.sbatch binomial
sbatch scripts/slurm/surya_train.sbatch copoc
```

Use the `last.ckpt` or a selected validation checkpoint produced in the matching
`assets/checkpoints/{qr,cls,copoc}` directory. Calibration deliberately requires
the checkpoint name explicitly:

```bash
export SURYA_CHECKPOINT='last.ckpt'
sbatch scripts/slurm/surya_calibrate.sbatch qr
```

For a train-to-calibrate dependency, capture the training job ID and use
`sbatch --dependency=afterok:<job-id> ...`. Add your cluster's account,
partition, and GPU-type directives to the top of each `.sbatch` file.

The checked-in calibration split is named `leaky_validation.csv`. Before making
reported results, verify it has no timestamp or active-region overlap with model
selection/validation data and no overlap with the test set. If it is the same
validation population used to choose the checkpoint, create and configure a
separate calibration split.
