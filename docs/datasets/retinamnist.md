---
dataset_id: retinamnist
name: RetinaMNIST
card_version: "0.2.0"
status: provisional
project: ordinal-conformal-prediction
method_compatibility:
  ocqr: "0.3.0"
representation_type: class_only_surrogate
input_modality:
  - retinal_fundus_image
numeric_target:
  symbol: Z
  source: ordinal_label_embedding
  transformation: "Z = Y_ord"
ordinal_label:
  symbol: Y_ord
  source: supplied_dataset_label
class_count: 5
bins:
  convention: "B_k = [b_k, b_{k+1})"
  threshold_equality: right_bin
  thresholds: [0.5, 1.5, 2.5, 3.5]
split_policy: "official MedMNIST train/validation/test; train split stratified 70/30 into training/calibration with seed 42"
license: "CC BY 4.0"
source_url: "https://zenodo.org/records/10519652/files/retinamnist.npz?download=1"
last_updated: "2026-08-03"
---

# RetinaMNIST Dataset Card and OCQR Metadata

## 1. Purpose

RetinaMNIST is used as a class-only ordinal image dataset for evaluating Ordinal Conformalized Quantile Regression (OCQR). Because the project documents do not define an observed continuous disease-severity target, the canonical interface uses a fixed class-index embedding.

This file is a dataset-specific contract. It supplements the OCQR theory and method contract and must be versioned with the code and experiment configuration.

## 2. Canonical OCQR interface

Each retained observation must expose

\[
(X,Z,Y_{\mathrm{ord}}),
\]

where:

- \(X\): retinal fundus image;
- \(Y_{\mathrm{ord}}\in\{0,\ldots,K-1\}\): supplied ordered retinal-severity label after any documented remapping;
- \(Z=Y_{\mathrm{ord}}\): canonical surrogate numeric coordinate used for pinball loss and the CQR score.

The implementation must still pass \(Z\) and \(Y_{\mathrm{ord}}\) as logically distinct fields.

## 3. Target representation

### 3.1 Canonical embedding

\[
e(k)=k,\qquad Z=e(Y_{\mathrm{ord}})=Y_{\mathrm{ord}}.
\]

This is a surrogate ordinal coordinate, not a measured continuous disease-severity value. Validity applies to this declared embedding. Prediction-set membership and efficiency are not guaranteed to remain unchanged under another monotone recoding.

### 3.2 Alternative embeddings

Any alternative embedding must be:

- assigned a distinct variant name;
- fixed before calibration data are accessed;
- documented with its own thresholds;
- reported as an ablation rather than mixed with canonical results.

## 4. Label mapping

| Canonical index | Source label | Clinical meaning | Retained? |
|---:|---|---|---|
| 0 | `0` | No diabetic retinopathy | Yes |
| 1 | `1` | Mild diabetic retinopathy | Yes |
| 2 | `2` | Moderate diabetic retinopathy | Yes |
| 3 | `3` | Severe diabetic retinopathy | Yes |
| 4 | `4` | Proliferative diabetic retinopathy | Yes |

The source-to-canonical mapping is the identity mapping supplied by the MedMNIST RetinaMNIST release.

## 5. Bin contract

For a class-index embedding with \(K\) classes, the intended canonical construction is midpoint binning:

\[
B_k=[b_k,b_{k+1}),
\]

with extended endpoints \(b_0=-\infty\), \(b_K=+\infty\), and internal thresholds at the midpoints between adjacent embedded class values.

For the canonical embedding \(e(k)=k\), the expected internal thresholds are

\[
b_j=j-\tfrac12,\qquad j=1,\ldots,K-1.
\]

The configured threshold array is `[0.5, 1.5, 2.5, 3.5]`.

Target and threshold comparisons must use a common floating-point dtype. Equality at an internal threshold belongs to the bin on the right.

## 6. Required consistency condition

Every retained observation must satisfy

\[
Y_{\mathrm{ord}}=k\Longleftrightarrow Z\in B_k.
\]

Because \(Z=Y_{\mathrm{ord}}\), this must be verified for every embedded class value and every threshold boundary.

## 7. Preprocessing and exclusions

The following fields must be completed from the implemented data pipeline:

- image size and channel format: 28 × 28 RGB `uint8` images in the reviewed NPZ;
- normalization: `ToTensor()` followed by channelwise normalization with mean `[0.5]` and standard deviation `[0.5]`;
- augmentation used for training only: none in the current adapter;
- invalid or unreadable image handling: NPZ-backed MedMNIST loading; no adapter-specific invalid-image recovery;
- label exclusions or remapping: none; labels `0` through `4` are retained unchanged;
- duplicate handling: not implemented by the adapter.

No preprocessing decision may depend on calibration or test performance.

## 8. Split and provenance

| Field | Value |
|---|---|
| Original split source | MedMNIST RetinaMNIST release: 1,080 train, 120 validation, 400 test samples |
| Train manifest/hash | Stratified 70% subset of official train split; not persisted as a standalone manifest |
| Validation manifest/hash | Official validation split contained in `retinamnist.npz` |
| Calibration manifest/hash | Stratified 30% subset of official train split; not persisted as a standalone manifest |
| Test manifest/hash | Official test split contained in `retinamnist.npz` |
| Split seed | 42 for train/calibration partition |
| Dataset release/version | Reviewed local `retinamnist.npz`, SHA-256 `254915f5f0a2074665c4676356824cf4ef4a3bcab233894b4bafcaf48962bd69` |
| Adapter version | Repository package `ordinal_cqr`, card version 0.2.0 |

Training and validation data may be used for model and method selection. Calibration data may enter canonical OCQR only through the prespecified class-specific correction calculation. Test data must remain untouched until final evaluation.

## 9. Automated validation requirements

The dataset adapter and tests must verify:

1. all retained labels map to consecutive canonical indices;
2. the label order is fixed and documented;
3. \(Z\) and \(Y_{\mathrm{ord}}\) are exposed separately;
4. \(Z=Y_{\mathrm{ord}}\) for every retained sample;
5. each embedded value falls into exactly one expected bin;
6. integer-valued targets and floating thresholds are compared without integer truncation;
7. equality at every internal threshold maps to the right bin;
8. no calibration grouping is recomputed from model predictions;
9. class counts are recorded for every split;
10. split and configuration hashes are written to experiment metadata.

## 10. Known limitations

- The numeric target is a surrogate class coordinate.
- Efficiency and raw candidate membership may depend on the chosen embedding.
- Train and calibration subsets are deterministic under the current package versions and seed, but their sample-ID manifests are not persisted.

## 11. Completion checklist

- [x] Confirm class count and ordered label meanings.
- [x] Freeze source-to-canonical label mapping.
- [x] Materialize midpoint thresholds.
- [x] Document preprocessing and exclusions.
- [x] Record dataset release, license, and source URL.
- [ ] Freeze train/validation/calibration/test manifests.
- [ ] Add target-label-bin consistency tests.
