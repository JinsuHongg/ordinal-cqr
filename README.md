# OCQR-Solar: Ordinal Conformalized Quantile Regression for Solar Flare Forecasting

![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)

OCQR-Solar is a specialized uncertainty quantification (UQ) and conformal prediction (CP) framework developed for high-stakes ordinal classification. The primary application of this architecture is space weather forecasting, specifically solar flare severity prediction.

## The Label Space

Solar flares exhibit a highly skewed, heavy-tailed distribution in severity. We model this physical phenomenon as a 5-class ordinal classification problem:

*   **Classes:** $K=5$ ordinal levels mapped to integers: `{"FQ/A": 0, "B": 1, "C": 2, "M": 3, "X": 4}`.
*   **Natural Ordering:** $0 < 1 < 2 < 3 < 4$.
*   **The Zero-Disjoint Axiom:** Prediction sets must never omit intermediate ordinal states. For instance, a prediction interval of `[0, 1, 2]` (FQ, B, C) is logically valid. Conversely, a disjoint set such as `[0, 4]` (FQ, X) or `[1, 3]` (B, M) physically violates domain constraints and represents a fatal algorithmic failure.

## Repository Structure

The repository is modularized into dedicated PyTorch Lightning components. Below is a high-level overview of the architectural structure:

```text
OCQR-Solar/
├── assets/                     # Persistent storage for model checkpoints (.ckpt), Wandb telemetry, and generated evaluation artifacts.
├── configs/                    # Hydra YAML configurations controlling hyperparameters for backbone models and conformal experiments.
│   ├── cls/                    # Configurations for baseline nominal classification backbones (ResNet).
│   └── qr/                     # Configurations for continuous quantile regression backbones (Pinball loss).
├── scripts/                    # Entry points for execution.
│   └── experiments/            # Scripts for initiating model training and executing conformal calibration loops.
└── src/ocqr_solar/             # Core Python package housing the primary logic.
    ├── datamodules/            # PyTorch Lightning DataModules (handles dynamic batching, cross-validation splits, and memory pinning).
    ├── datasets/               # Native PyTorch Dataset classes handling disk I/O for Adience, Retina-MNIST, and Solar Flare tensors.
    ├── explainability/         # Implementation of Mondrian conformal score computations and quantile thresholding operations.
    ├── metrics/                # Vectorized evaluators for contiguity (SFS, MDJ, CCR) and probability density.
    ├── models/                 # Neural architectures including base regressors, classifiers, and Lightning Module wrappers.
    └── utils/                  # Telemetry hooks, callback definitions, and helper functions.
```

## Core Methodology: Ordinal CQR

OCQR combines continuous quantile regression, true-class Mondrian calibration, and an ordinal closure operation. The separation between the continuous target domain and the discrete reporting labels is essential to the method.

### 1. Numeric Target Policy

The quantile-regression target $Y$ must be a numeric coordinate in the ordered domain. When an underlying measurement is available, such as flare magnitude or age in years, OCQR uses that measurement. When a dataset provides only ordinal classes, OCQR uses a documented ordinal index embedding such as $0,\ldots,K-1$ with midpoint thresholds. An index embedding is a modeling convention, not a claim that the class IDs are physical continuous measurements, and it must be reported with the results.

For $K$ classes, let the strictly increasing internal thresholds be $\tau_1 < \dots < \tau_{K-1}$. They define the bins

$$
B_0=(-\infty,\tau_1),\quad
B_k=[\tau_k,\tau_{k+1})\ (1\leq k<K-1),\quad
B_{K-1}=[\tau_{K-1},\infty).
$$

Threshold equality is assigned to the bin on the right. The same target policy and thresholds must be used for training, calibration, and evaluation.

### 2. Pinball Quantile Training

The QR backbone predicts lower and upper conditional quantiles $L(X)$ and $U(X)$, with an optional median output. For quantile level $r$, training minimizes the pinball loss

$$
\ell_r(y,\hat{y})=\max\{r(y-\hat{y}),(r-1)(y-\hat{y})\}.
$$

The implementation orders the two endpoint predictions before calibration and inference, so accidental quantile crossing cannot produce a reversed base interval. This runtime safeguard does not replace monitoring or penalizing the quantile-crossing rate during model development.

### 3. True-Bin Mondrian Calibration

For each calibration example, OCQR computes

$$
s_i=\max\{L(X_i)-Y_i,\;Y_i-U(X_i)\}
$$

and assigns the score to the bin containing the true numeric target, $k_i$ such that $Y_i\in B_{k_i}$. For a class with $n_k>0$ calibration samples, define

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

This candidate-wise inversion is the link between true-bin Mondrian calibration and inference: the true candidate is evaluated using the correction calibrated for its own bin. All candidate intervals and bin-overlap tests are computed as broadcast PyTorch tensors.

### 5. Ordinal Hull and Safe Fallbacks

Candidate-specific corrections can produce a fragmented raw set. OCQR therefore returns its ordinal hull

$$
C(X)=\{\min S(X),\min S(X)+1,\dots,\max S(X)\}.
$$

The hull only adds labels, so it cannot reduce coverage, and every non-empty returned set is contiguous by construction. If signed corrections or numerical edge cases produce an empty raw set, the implementation conservatively returns the full ordinal label space.

A class with no calibration examples cannot support an empirical class-conditional quantile. OCQR assigns that unsupported candidate an infinite correction, which includes it conservatively, records its unsupported status, and avoids the anti-conservative behavior of silently using zero. Results must report per-class calibration counts; an infinite fallback is a safety policy, not evidence of an estimated conditional guarantee for that class.

### Guarantees and Assumptions

For every class with an attainable finite-sample rank, split-conformal exchangeability within that class and an exact order statistic give the usual Mondrian coverage statement for the numeric target. Numeric-target coverage implies inclusion of its true bin under the candidate-overlap rule. Ordinal hull closure preserves that inclusion while guaranteeing zero disjoint gaps. If the requested rank exceeds $n_k$, the nominal level is unattainable from that class's empirical sample alone; the implementation uses an infinite correction rather than presenting the largest observed score as a valid nominal quantile.

These guarantees require fixed training/calibration/test splits, thresholds and target mappings chosen without test-set feedback, and calibration examples exchangeable with future examples within each reported class. They do not imply conditional coverage for unsupported classes, robustness to distribution shift, or validity after test-driven checkpoint, threshold, or hyperparameter selection. Small rare-class calibration counts can also make the valid correction highly conservative.

## UQ Method Comparison

The baselines use different model outputs and target different statistical objectives. Comparisons should therefore report the guarantee each implementation actually targets, along with marginal coverage, per-class coverage, worst-class coverage, set size, SFS, MDJ, and CCR.

| Method | Required backbone | Calibration target in this repository | Contiguity mechanism | Class-wise support | Principal tradeoff |
|---|---|---|---|---|---|
| **OCQR** | Continuous quantile regression | Exact marginal or true-bin Mondrian CQR | Candidate-specific bin overlap followed by ordinal hull | Yes | Uses ordered numeric geometry and supports class-conditional analysis, but rare-class corrections and hull filling can enlarge sets. |
| **OAPS** | Standard Softmax classifier | APS-style probability-mass conformity | Ordinal probability construction | Heuristic class-wise option | Does not require numeric targets, but the current predicted-class threshold selection is not a true-label Mondrian guarantee. |
| **min-CPS** | Standard Softmax classifier | Pooled shortest contiguous probability-mass heuristic | Exhaustive contiguous interval search | No | Directly favors short intervals; the current selection rule should be evaluated empirically and is not documented here as a Mondrian guarantee. |
| **min-RCPS** | Standard Softmax classifier | Pooled probability-mass objective with a length penalty | Regularized contiguous interval search | No | Trades interval mass against width through a tuning parameter; the current implementation is not class-conditional. |
| **Risk control** | Standard Softmax classifier | Pooled expected ordinal distance to the returned set | Probability thresholding followed by hull filling | No | Targets ordinal-distance risk rather than label coverage; finite-sample claims depend on the validity of the threshold-selection bound. |
| **COPOC** | Unimodal classifier, such as the binomial head | Pooled LAC score $1-p_Y$ | A superlevel set of a verified unimodal distribution | No | Gives contiguous level sets without hull filling, but relies on a correctly constrained unimodal probability model. |

OAPS, min-CPS, min-RCPS, and risk control must use a standard Softmax classification backbone. COPOC must use the unimodal/binomial backbone. OCQR must use the quantile-regression backbone with either a physical numeric target or a documented ordinal index embedding. Using one checkpoint architecture for every method is not a valid comparison.

## Evaluation Metrics

Standard conformal prediction metrics, such as marginal coverage and set size, fail to penalize disjoint anomalies. OCQR-Solar evaluates model integrity using structural metrics:

*   **CCR (Contiguous Coverage Rate):** The primary benchmark metric. Defines the proportion of samples where the prediction set is strictly contiguous and successfully captures the ground truth label $Y$.
*   **SFS (Set Fragmentation Score):** Quantifies the number of disconnected sub-segments within a prediction set. The target value is exactly `1.0`. Any value `> 1.0` indicates a fundamental contiguity violation.
*   **MDJ (Maximum Disjoint Jump):** Quantifies the maximum magnitude of omitted intermediate classes (e.g., predicting `[1, 4]` produces an MDJ of `2`). The target value is `0.0`.

## Installation & Execution

### 1. Environment Initialization

Dependencies are rigidly managed via conda/mamba.
```bash
conda env create -f environment.yml
conda activate ocqr_solar
```

### 2. Available Benchmark Datasets

The repository supports multiple distinct DataModules to facilitate rigorous unit testing and ablation studies:
- **`FlareSuryaBench`**: The primary operational space-weather dataset of solar flare image sequences.
- **`Retina-MNIST`**: 5-class ordinal medical imaging benchmark for accelerated local algorithm validation.
- **`Adience`**: 8-class biological dataset emphasizing continuous physical quantity estimation.

### 3. Model Training

Initiate training for a chosen architecture using Hydra configuration overrides:
```bash
# Train the baseline ordinal classification model
python scripts/experiments/training.py --config-path ../../configs/cls --config-name CLS_resnet18_binomial_train_adience

# Train the continuous quantile regression backbone
python scripts/experiments/training.py --config-path ../../configs/qr --config-name QR_resnet18_train_adience
```

### 4. Conformal Calibration Validation

Following model convergence, extract calibrated prediction sets and analyze structural contiguity:
```bash
python scripts/experiments/calibration.py --config-path ../../configs/qr --config-name QR_resnet18_calibration_adience
```
