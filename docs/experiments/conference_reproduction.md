# Conference v0.3 reproduction

Run commands from the repository root with the existing `ocqr_solar`
interpreter. In this execution workspace, `conda run` rejects its externally
mounted environment directory because it is not writable to the sandbox; the
interpreter itself is usable. No command may reuse an existing run directory.

```bash
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/training.py \
  --config-name experiments/conference_v0_3/retinamnist seed=0
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/calibration.py \
  --config-name experiments/conference_v0_3/retinamnist seed=0
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/synthetic_ocqr_validation.py \
  --output results/conference_v0_3/synthetic_validation.json
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/build_utkface_manifest.py
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/run_conference_experiment.py \
  --config configs/experiments/conference_v0_3/retinamnist.yaml --method ocqr --seed 0
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/run_conference_experiment.py \
  --config configs/experiments/conference_v0_3/retinamnist.yaml --method lac --seed 0
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/run_conference_experiment.py \
  --config configs/experiments/conference_v0_3/retinamnist.yaml --method aps --seed 0
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/run_conference_experiment.py \
  --config configs/experiments/conference_v0_3/retinamnist.yaml --method oaps --seed 0
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/run_conference_experiment.py \
  --config configs/experiments/conference_v0_3/retinamnist.yaml --method copoc --seed 0
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python scripts/experiments/aggregate_conference_results.py \
  --runs outputs/conference_v0_3 --results results/conference_v0_3
PYTHONPATH=src /mnt/storage/conda_envs/ocqr_solar/bin/python -m pytest -q
```

The UTKFace manifest command freezes the sorted-filename, seed-0,
stratified 60/10/20/10 membership at
`data/manifests/conference_v0_3/utkface/manifest.jsonl`. It refuses to
overwrite an existing manifest unless `--overwrite` is supplied explicitly.

The first two legacy entry points remain available, but canonical RetinaMNIST
and UTKFace runs use `run_conference_experiment.py`. Replace the dataset config,
method, and seed explicitly for each run; existing output directories are
rejected unless `--overwrite` is deliberately supplied. Solar runs require the
frozen chronological manifest, leakage audit, portable paths, and validation-
selected checkpoint documented in the conference plan before aggregation.

APS artifacts created before protocol version
`1.0.0-nonrandomized-boundary`, LAC artifacts without
`1.0.0-exact-split` provenance, old OAPS prefix artifacts, and legacy
binomial-LAC COPOC artifacts are rejected rather than mixed into tables.

Aggregation refuses legacy results and generates `main_results.csv`,
`per_class_results.csv`, `ablation_results.csv`, `calibration_diagnostics.csv`,
and LaTex tables. Figures require `matplotlib`, which is already expected in
the declared environment through its Lightning stack; a missing import is a
visible failure, not a skipped figure presented as complete.
