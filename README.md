# Ordinal CQR: Class-Conditional Conformal Prediction with Contiguous Ordinal Sets

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)

Ordinal Conformalized Quantile Regression (OCQR) is an uncertainty-quantification and conformal-prediction framework for ordinal outcomes. It combines quantile regression, true-label Mondrian calibration, candidate-specific score inversion, and ordinal hull closure to produce nonempty contiguous prediction sets. The repository supports ordinal image datasets with either numeric targets and ordinal labels or ordinal labels alone.

## The Label Space

Let the ordered label space be

$$
\mathcal Y=\{0,1,\ldots,K-1\},\qquad 0<1<\cdots<K-1.
$$

Final prediction sets must respect this order and never omit intermediate labels. For example, `{0, 1, 2}` is contiguous, whereas `{0, 2}` and `{1, 3}` are not valid final OCQR outputs. Raw candidate sets may be fragmented; the ordinal hull resolves those gaps explicitly.

## Repository Structure

The repository is modularized into dedicated PyTorch Lightning components. Below is a high-level overview of the architectural structure:

```text
ordinal-cqr/
├── assets/                     # Local, ignored checkpoints, telemetry, and generated evaluation artifacts.
├── configs/                    # Hydra YAML configurations controlling hyperparameters for backbone models and conformal experiments.
│   ├── cls/                    # Configurations for baseline nominal classification backbones (ResNet).
│   └── qr/                     # Configurations for continuous quantile regression backbones (Pinball loss).
├── scripts/                    # Entry points for execution.
│   └── experiments/            # Scripts for initiating model training and executing conformal calibration loops.
├── docs/methods/               # Normative OCQR contract and theoretical statement.
├── tests/                      # Focused method, data-interface, metadata, and metric tests.
└── src/ordinal_cqr/            # Core Python package housing the primary logic.
    ├── datamodules/            # PyTorch Lightning DataModules and dataset split construction.
    ├── datasets/               # Dataset adapters for the supported ordinal benchmarks.
    ├── explainability/         # Implementation of Mondrian conformal score computations and quantile thresholding operations.
    ├── metrics/                # Vectorized classification, coverage, set-size, and contiguity metrics.
    ├── models/                 # Neural architectures including base regressors, classifiers, and Lightning Module wrappers.
    └── utils/                  # Telemetry hooks, callback definitions, and helper functions.
```

## Core Methodology: Ordinal CQR

OCQR combines continuous quantile regression, true-class Mondrian calibration, and an ordinal closure operation. The separation between the continuous target domain and the discrete reporting labels is essential to the method.

### 1. Numeric Target Policy

Each sample exposes $(X,Z,Y_{\mathrm{ord}})$, where $Z$ is the numeric quantile-regression target and $Y_{\mathrm{ord}}$ is the supplied ordinal label. When an underlying measurement is available, such as age in years, OCQR uses that measurement as $Z$. When a dataset provides only ordinal classes, it uses the documented class-index embedding $Z=Y_{\mathrm{ord}}$, typically with midpoint thresholds. An index embedding is a modeling convention, not a claim that class IDs are physical continuous measurements, and it must be reported with the results.

For $K$ classes, let the strictly increasing internal thresholds be $\tau_1 < \dots < \tau_{K-1}$. They define the bins

- $B_0 = [-\infty, \tau_1)$;
- $B_k = [\tau_k, \tau_{k+1})$ for $1 \leq k < K-1$;
- $B_{K-1} = [\tau_{K-1}, \infty)$.

Threshold equality is assigned to the bin on the right. The same target policy and thresholds must be used for training, calibration, and evaluation, and every retained sample must satisfy $Z\in B_{Y_{\mathrm{ord}}}$.

### 2. Pinball Quantile Training

The QR backbone predicts lower and upper conditional quantiles $L(X)$ and $U(X)$, with an optional median output. For quantile level $r$, training minimizes the pinball loss

$$
\ell_r(y,\hat{y})=\max\{r(y-\hat{y}),(r-1)(y-\hat{y})\}.
$$

The implementation orders the two endpoint predictions before calibration and inference, so accidental quantile crossing cannot produce a reversed base interval. This runtime safeguard does not replace monitoring or penalizing the quantile-crossing rate during model development. Canonical v0.3 uses one deterministic evaluation-mode QR forward pass; Monte Carlo dropout would be a separate method variant requiring its own prespecified endpoint construction and calibration.

### 3. True-Label Mondrian Calibration

For each calibration example, OCQR computes

$$
s_i=\max\{L(X_i)-Z_i,\;Z_i-U(X_i)\}
$$

and assigns it to the supplied true ordinal class $Y_{\mathrm{ord},i}$. The implementation separately validates $Z_i\in B_{Y_{\mathrm{ord},i}}$; it does not derive Mondrian groups from model predictions or silently replace the supplied class with a target-derived label. For a class with $n_k>0$ calibration samples, define

$$
r_k=\left\lceil(n_k+1)(1-\alpha)\right\rceil.
$$

When $r_k\leq n_k$, the correction $\hat q_k$ is the $r_k$-th order statistic of that class's scores. The implementation uses this exact finite-sample order statistic, not an interpolated numeric quantile. When $r_k>n_k$, it uses the conservative infinite correction described below. Marginal mode remains available and applies the same construction to all calibration scores with one shared $\hat q$.

### 4. Candidate-Specific Membership

The true class is unknown at inference, so OCQR does not select a correction using a point prediction. Instead, every class $k$ is evaluated as a candidate with its own interval

$$
I_k(X)=[L(X)-\hat q_k,\;U(X)+\hat q_k].
$$

The raw candidate set is

$$
S(X)=\{k:I_k(X)\cap B_k\neq\varnothing\}.
$$

This candidate-wise inversion is the link between true-label Mondrian calibration and inference: the true candidate is evaluated using the correction calibrated for its own class. All candidate intervals and bin-overlap tests are computed as broadcast PyTorch tensors.

### 5. Ordinal Hull and Safe Fallbacks

Candidate-specific corrections can produce a fragmented or empty raw set. Define the conservative fallback

$$
\widetilde S(X)=
\begin{cases}
S(X), & S(X)\neq\varnothing,\\
\{0,\ldots,K-1\}, & S(X)=\varnothing.
\end{cases}
$$

OCQR returns the ordinal hull

$$
C(X)=\{\min \widetilde S(X),\min \widetilde S(X)+1,\dots,\max \widetilde S(X)\}.
$$

Fallback and hull closure only add labels, so they cannot reduce coverage, and every final set is nonempty and contiguous. Empty raw sets can arise from finite signed corrections. Nonfinite model endpoints or targets are rejected explicitly rather than handled by fallback.

A class with no calibration examples cannot support an empirical finite class-conditional quantile. OCQR assigns that candidate an infinite correction, which includes it conservatively, records its unsupported status, and avoids silently using an anti-conservative zero correction. This gives coverage one for that candidate before add-only post-processing, but no informative finite correction or useful efficiency claim. Results must report per-class counts, requested ranks, rank attainability, corrections, and support status.

### Guarantees and Assumptions

For every class with an attainable finite-sample rank, split-conformal exchangeability within that class and an exact order statistic give the usual Mondrian coverage statement for the numeric target. Numeric-target coverage implies inclusion of its true bin under the candidate-overlap rule. Ordinal hull closure preserves that inclusion while guaranteeing contiguous, nonempty final sets. If the requested rank exceeds $n_k$, the nominal finite empirical correction is unattainable; the implementation uses an infinite correction rather than presenting the largest observed score as a valid nominal quantile.

These guarantees require a model and representation fixed independently of calibration outcomes, fixed thresholds and target mappings, disjoint calibration/test data, and calibration examples exchangeable with future examples within each reported class. They do not establish exchangeability, robustness to distribution shift, or validity after calibration- or test-driven checkpoint, threshold, or hyperparameter selection. Dataset shift, temporal extrapolation, and dependent observations therefore require separate empirical analysis. Small rare-class calibration counts can also make valid prediction sets highly conservative.

The normative method definition and proof assumptions are documented in [`docs/methods/ocqr_contract.md`](docs/methods/ocqr_contract.md) and [`docs/methods/ocqr_theory.md`](docs/methods/ocqr_theory.md).

## UQ Method Comparison

The baselines use different model outputs and target different statistical objectives. Comparisons should therefore report the guarantee each implementation actually targets, along with marginal coverage, per-class coverage, worst-class coverage, set size, SFS, MDJ, and CCR.

| Method | Required backbone | Calibration target in this repository | Contiguity mechanism | Class-wise support | Principal tradeoff |
|---|---|---|---|---|---|
| **OCQR** | Continuous quantile regression | Exact marginal or true-label Mondrian CQR | Candidate-specific bin overlap followed by ordinal hull | Yes | Uses ordered numeric geometry and supports class-conditional analysis, but rare-class corrections and hull filling can enlarge sets. |
| **OAPS** | Standard Softmax classifier | APS-style probability-mass conformity | Ordinal probability construction | Heuristic class-wise option | Does not require numeric targets, but the current predicted-class threshold selection is not a true-label Mondrian guarantee. |
| **min-CPS** | Standard Softmax classifier | Pooled shortest contiguous probability-mass heuristic | Exhaustive contiguous interval search | No | Directly favors short intervals; the current selection rule should be evaluated empirically and is not documented here as a Mondrian guarantee. |
| **min-RCPS** | Standard Softmax classifier | Pooled probability-mass objective with a length penalty | Regularized contiguous interval search | No | Trades interval mass against width through a tuning parameter; the current implementation is not class-conditional. |
| **Risk control** | Standard Softmax classifier | Pooled expected ordinal distance to the returned set | Probability thresholding followed by hull filling | No | Targets ordinal-distance risk rather than label coverage; finite-sample claims depend on the validity of the threshold-selection bound. |
| **COPOC** | Unimodal classifier, such as the binomial head | Pooled LAC score $1-p_Y$ | A superlevel set of a verified unimodal distribution | No | Gives contiguous level sets without hull filling, but relies on a correctly constrained unimodal probability model. |

OAPS, min-CPS, min-RCPS, and risk control must use a standard Softmax classification backbone. COPOC must use the unimodal/binomial backbone. OCQR must use the quantile-regression backbone with either a physical numeric target or a documented ordinal index embedding. Using one checkpoint architecture for every method is not a valid comparison.

## Evaluation Metrics

Standard conformal prediction metrics, such as marginal coverage and set size, do not reveal disjoint anomalies. Ordinal CQR therefore also evaluates structural metrics. The following targets apply to final nonempty prediction sets; raw candidate sets are intentionally allowed to be empty or fragmented before fallback and hull closure.

*   **CCR (Contiguous Coverage Rate):** The proportion of samples where the prediction set is strictly contiguous and contains the ground-truth ordinal label $Y_{\mathrm{ord}}$.
*   **SFS (Set Fragmentation Score):** Quantifies the number of disconnected sub-segments within a prediction set. The target value is exactly `1.0`. Any value `> 1.0` indicates a fundamental contiguity violation.
*   **MDJ (Maximum Disjoint Jump):** Quantifies the maximum magnitude of omitted intermediate classes (e.g., predicting `[1, 4]` produces an MDJ of `2`). The target value is `0.0`.

Evaluation artifacts additionally report raw coverage and set size, raw-empty and fragmentation rates, fallback/hull/total inflation, full-set rate, and per-class coverage and set size.

## Installation & Execution

### 1. Environment Initialization

Dependencies are declared in the checked-in Conda environment. Its legacy environment name is retained for compatibility with existing machines and checkpoints.

```bash
conda env create -f environment.yml
conda activate ocqr_solar
```

### 2. Available Benchmark Datasets

The repository supports multiple distinct DataModules to facilitate rigorous unit testing and ablation studies:
- **`Retina-MNIST`**: 5-class ordinal medical imaging benchmark for accelerated local algorithm validation.
- **`UTKFace`**: Age regression with exact age as $Z$ and configured age bins as $Y_{\mathrm{ord}}$.
- **`Adience`**: 8-class age benchmark using documented bin representatives when exact age is unavailable.
- **`EyePACS`**: 5-class diabetic-retinopathy benchmark using the class-index embedding.
- **`FlareSuryaBench`**: 5-class space-weather benchmark with a continuous intensity target and an ordinal event label.

### 3. Model Training

Initiate training for a chosen architecture using Hydra configuration overrides:
```bash
# Train a standard Softmax classification backbone for applicable baselines
PYTHONPATH=src python scripts/experiments/training.py --config-path ../../configs/cls --config-name CLS_resnet18_train_adience

# Train the continuous quantile regression backbone
PYTHONPATH=src python scripts/experiments/training.py --config-path ../../configs/qr --config-name QR_resnet18_train_adience
```

### 4. Conformal Calibration Validation

Following model convergence, extract calibrated prediction sets and analyze structural contiguity:
```bash
PYTHONPATH=src python scripts/experiments/calibration.py --config-path ../../configs/qr --config-name QR_resnet18_calibration_adience
```

Before running an experiment, configure its dataset paths and set its checkpoint entry to an existing trained checkpoint. Verify that its thresholds match the checkpoint's training target. Canonical OCQR runs require `uc.class_wise: true` and a calibration split distinct from the test split.

The calibration command writes linked strict-JSON artifacts under `uc.csv_path`: per-class calibration metadata and evaluation metrics including raw-set, fallback, hull, coverage, set-size, SFS, MDJ, and CCR diagnostics. It also writes per-sample CSV predictions. Current artifact export is intended for the single-device configurations checked into `configs/qr/`; distributed CSV gathering is not yet implemented.
