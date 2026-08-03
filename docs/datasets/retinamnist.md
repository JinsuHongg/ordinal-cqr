---
dataset_id: retinamnist
name: RetinaMNIST
card_version: "0.1.0"
status: draft
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
class_count: TBD
bins:
  convention: "B_k = [b_k, b_{k+1})"
  threshold_equality: right_bin
  thresholds: TBD_midpoint_thresholds
split_policy: TBD
license: TBD
source_url: TBD
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
| 0 | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD |
| 4 or higher | TBD | TBD | TBD |

The exact class count, official ordering, and source-to-canonical mapping are not specified in the supplied OCQR documents and must be completed from the dataset adapter or authoritative dataset documentation.

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

The final threshold array must be materialized in configuration or metadata after \(K\) is confirmed.

Target and threshold comparisons must use a common floating-point dtype. Equality at an internal threshold belongs to the bin on the right.

## 6. Required consistency condition

Every retained observation must satisfy

\[
Y_{\mathrm{ord}}=k\Longleftrightarrow Z\in B_k.
\]

Because \(Z=Y_{\mathrm{ord}}\), this must be verified for every embedded class value and every threshold boundary.

## 7. Preprocessing and exclusions

The following fields must be completed from the implemented data pipeline:

- image size and channel format: `TBD`;
- normalization: `TBD`;
- augmentation used for training only: `TBD`;
- invalid or unreadable image handling: `TBD`;
- label exclusions or remapping: `TBD`;
- duplicate handling: `TBD`.

No preprocessing decision may depend on calibration or test performance.

## 8. Split and provenance

| Field | Value |
|---|---|
| Original split source | TBD |
| Train manifest/hash | TBD |
| Validation manifest/hash | TBD |
| Calibration manifest/hash | TBD |
| Test manifest/hash | TBD |
| Split seed | TBD |
| Dataset release/version | TBD |
| Adapter version | TBD |

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
- The supplied project documents do not establish the exact source-label meanings, class count, preprocessing, license, or split hashes.

## 11. Completion checklist

- [ ] Confirm class count and ordered label meanings.
- [ ] Freeze source-to-canonical label mapping.
- [ ] Materialize midpoint thresholds.
- [ ] Document preprocessing and exclusions.
- [ ] Record dataset release, license, and source URL.
- [ ] Freeze train/validation/calibration/test manifests.
- [ ] Add target-label-bin consistency tests.
