# Ordinal CQR: Class-Conditional Conformal Prediction with Contiguous Ordinal Sets

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-red.svg)

Ordinal Conformalized Quantile Regression (OCQR) is a conformal prediction framework for ordinal outcomes $\mathcal{Y} = \{0, 1, \ldots, K-1\}$. It combines continuous quantile regression, true-label Mondrian calibration, candidate-specific interval-bin intersection, and an add-only ordinal hull to produce nonempty, contiguous prediction sets with finite-sample class-conditional coverage under within-class exchangeability.

## Repository Structure

```text
ordinal-cqr/
├── configs/
│   ├── method/ocqr.yaml          # Canonical OCQR v0.3.0 policy settings
│   ├── qr/                       # Hydra configs for quantile-regression training & OCQR calibration
│   └── cls/                      # Hydra configs for classification-backbone baselines
├── scripts/
│   └── experiments/
│       ├── training.py           # Unified training entrypoint (qr, cls, mcd)
│       ├── calibration.py        # Regression/QR UQ calibration (ordinal_cqr, cqr, cp, lp, mcd)
│       ├── ordinal_calibration.py# Classifier UQ baselines (lac, aps, oaps, copoc, min_cps, min_rcps, risk_control)
│       └── synthetic_ocqr_validation.py # Deterministic OCQR contract smoke validation
├── src/
│   ├── ordinal_cqr/              # Installable Python package
│   │   ├── datamodules/          # PyTorch Lightning DataModules & config factory
│   │   ├── datasets/             # Dataset adapters (RetinaMNIST, UTKFace, Adience, EyePACS, SuryaBench)
│   │   ├── explainability/       # OrdinalCQRWrapper and conformal baseline wrappers
│   │   ├── experiments/          # Shared prediction artifact evaluation helpers
│   │   ├── metrics/              # Coverage, set-size, CCR, SFS, and MDJ metrics
│   │   ├── models/               # ResNet backbones and Lightning modules
│   │   └── utils/                # Pinball loss, callbacks, WandB logging, dataset utilities
│   └── ocqr_solar/               # Backward-compatible namespace for legacy checkpoints
├── tests/                        # Unit and contract tests
├── environment.yml               # Conda environment specification (env name: ocqr)
└── pyproject.toml                # PEP 621 package configuration
```

## How OCQR Works

Each sample exposes $(X, Z, Y_{\mathrm{ord}})$, where $Z \in \mathbb{R}$ is a numeric target (an observed measurement such as age/flux, or the class-index embedding $Z = Y_{\mathrm{ord}}$ when only ordinal labels exist) and $B_0, \ldots, B_{K-1}$ partition $\mathbb{R}$ via strictly increasing thresholds $\tau_1 < \cdots < \tau_{K-1}$ with right-bin equality ($B_k = [\tau_k, \tau_{k+1})$):

1. **Quantile Regression & Crossing Correction**: Predict lower and upper quantiles and sort endpoints $\widehat\ell(X) \le \widehat u(X)$.
2. **True-Label Mondrian Calibration**: Compute signed CQR scores $s_i = \max(\widehat\ell(X_i) - Z_i, Z_i - \widehat u(X_i))$ grouped by supplied $Y_{\mathrm{ord}, i}$, using the exact augmented order statistic at rank $r_k = \lceil (N_k + 1)(1 - \alpha) \rceil$ (falling back to $+\infty$ when unattainable).
3. **Candidate-Specific Inversion**: Include class $k$ in the raw set $C_{\mathrm{raw}}(X)$ iff $[\widehat\ell(X) - \hat q_k, \widehat u(X) + \hat q_k] \cap B_k \neq \varnothing$.
4. **Fallback & Ordinal Hull**: Replace an empty raw set with $\mathcal{Y}$, then take $\operatorname{Hull}(\widetilde C_{\mathrm{raw}}(X)) = \{\min, \ldots, \max\}$ so final sets are always nonempty and contiguous.

### Evaluation Metrics

In addition to marginal, macro, and per-class coverage and set size, `OrdinalCQRMetrics` reports structural contiguity diagnostics:

- **CCR (Contiguous Coverage Rate)**: Fraction of samples whose prediction set is contiguous and contains $Y_{\mathrm{ord}}$.
- **SFS (Set Fragmentation Score)**: Number of disconnected ordinal segments (`1.0` when contiguous).
- **MDJ (Maximum Disjoint Jump)**: Largest gap of omitted intermediate classes (`0.0` when contiguous).

## Installation

Create the Conda environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate ocqr
pip install -e .
```

For running tests locally, install with dev extras (or use the `pytest` included in `environment.yml`):

```bash
pip install -e ".[dev]"
```

## Quickstart (Python API)

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from ordinal_cqr.explainability import OrdinalCQRWrapper

# Assume quantile_model(x) returns shape (batch, >=2) with [lower, ..., upper]
wrapper = OrdinalCQRWrapper(
    trained_model=quantile_model,
    num_classes=5,
    class_mapping={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
    thresholds=[0.5, 1.5, 2.5, 3.5],
    alpha=0.1,
    class_wise=True,
)

# Calibration batches must yield (X, Z, Y_ord)
cal_loader = DataLoader(TensorDataset(x_cal, z_cal, y_cal), batch_size=64)
wrapper.calibrate(cal_loader)

# Inspect per-class counts, ranks, and corrections
metadata = wrapper.get_calibration_metadata()

# Predict nonempty contiguous ordinal sets
out = wrapper.predict_step((x_test, z_test, y_test), batch_idx=0)
prediction_sets = out["prediction_set"]      # bool tensor of shape (N, K)
raw_sets = out["raw_prediction_set"]         # pre-hull candidate sets
```

## Running Experiments

### 1. Deterministic Contract Validation

Verify exact finite-sample rank, crossing correction, negative scores, empty-set fallback, and ordinal hull behavior on synthetic data:

```bash
python scripts/experiments/synthetic_ocqr_validation.py
```

### 2. Model Training

Supported datasets in `configs/qr/` and `configs/cls/` include `retinamnist`, `utkface`, `adience`, `eyepacs`, and `surya_bench`. Pass your local dataset path via Hydra overrides (e.g. `+data.data_dir=/path/to/data`):

```bash
# Train a quantile regression backbone for OCQR
python scripts/experiments/training.py \
  --config-path ../../configs/qr \
  --config-name QR_resnet18_train_retinamnist \
  +data.data_dir=./data/medmnist

# Train a classification backbone for baseline methods
python scripts/experiments/training.py \
  --config-path ../../configs/cls \
  --config-name CLS_resnet18_train_retinamnist \
  +data.data_dir=./data/medmnist
```

Add `wandb.enabled=false` (or keep the default `wandb.offline=true`) to run without an online Weights & Biases account.

### 3. Conformal Calibration & Evaluation

```bash
# OCQR / CQR / CP / LP calibration on a trained QR or MCD checkpoint
python scripts/experiments/calibration.py \
  --config-path ../../configs/qr \
  --config-name QR_resnet18_calibration_retinamnist \
  +data.data_dir=./data/medmnist \
  check_point.base=./assets/checkpoints/qr/ \
  check_point.resnet18_qr=YOUR_CHECKPOINT.ckpt

# Classification UQ baselines (lac, aps, oaps, min_cps, min_rcps, copoc, risk_control)
python scripts/experiments/ordinal_calibration.py \
  --config-path ../../configs/cls \
  --config-name CLS_resnet18_calibration_retinamnist \
  +data.data_dir=./data/medmnist \
  check_point.base=./assets/checkpoints/cls/ \
  check_point.resnet18_cls=YOUR_CHECKPOINT.ckpt
```

## Running Tests

```bash
pytest -q
```
