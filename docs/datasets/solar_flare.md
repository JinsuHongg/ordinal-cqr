---
dataset_id: solar_flare
name: Solar Flare Prediction Dataset
card_version: "0.1.0"
status: draft
project: ordinal-conformal-prediction
method_compatibility:
  ocqr: "0.3.0"
representation_type: observed_numeric_target
input_modality:
  - solar_active_region_observations
  - optional_multimodal_inputs_TBD
numeric_target:
  symbol: Z
  source: peak_xray_flux
  transformation: TBD_fixed_deterministic_transformation
  unit: TBD
ordinal_label:
  symbol: Y_ord
  source: supplied_flare_class
class_count: TBD
bins:
  convention: "B_k = [b_k, b_{k+1})"
  threshold_equality: right_bin
  thresholds: TBD_A_B_C_M_X_mapping
split_policy: chronological
license: TBD
source_url: TBD
last_updated: "2026-08-03"
---

# Solar Flare Prediction Dataset Card and OCQR Metadata

## 1. Purpose

The solar flare dataset is the principal domain application for OCQR. It evaluates ordinal uncertainty quantification under severe class imbalance and chronological distribution shift.

The numeric target is peak X-ray flux or a fixed deterministic transformation. The supplied flare class is the ordinal label used for Mondrian calibration. A reproducible validation artifact must establish target-label-bin consistency before canonical experiments.

## 2. Canonical OCQR interface

Each retained observation must expose

\[
(X,Z,Y_{\mathrm{ord}}),
\]

where:

- \(X\): solar active-region input or a frozen multimodal observational representation;
- \(Z\): peak X-ray flux or its fixed deterministic transformation;
- \(Y_{\mathrm{ord}}\): supplied ordered flare class after documented canonical mapping.

Calibration grouping must use the supplied true flare class, not a class predicted by the model and not a class recomputed from \(Z\), unless exact equivalence is validated and a derived-label interface is explicitly declared.

## 3. Prediction unit and task definition

| Field | Value |
|---|---|
| Prediction unit | TBD: active-region time, sample window, or event |
| Forecast horizon | TBD |
| Observation window | TBD |
| Target event selection | TBD |
| Multiple events in horizon | TBD aggregation rule |
| No-flare handling | TBD |
| Active-region identifier | TBD source field |
| Timestamp convention | TBD |

These decisions affect dependence, labels, sample counts, and reproducibility and must be frozen before calibration.

## 4. Numeric target

| Field | Value |
|---|---|
| Source quantity | Peak X-ray flux |
| Source instrument/catalog | TBD |
| Native unit | TBD |
| Canonical transformation | TBD |
| Nonpositive/missing values | TBD |
| Finite-value requirement | Required |
| Precision and dtype | Floating point |

If a logarithmic or other deterministic transformation is used, all thresholds must be represented in the transformed coordinate. The transformation cannot be selected using calibration or test outcomes.

## 5. Ordinal label taxonomy

The intended domain ordering includes flare-severity classes such as A, B, C, M, and X, but the exact retained label space remains to be finalized.

| Canonical index | Supplied class | Meaning | Retained? | Notes |
|---:|---|---|---|---|
| 0 | TBD | Lowest retained severity or no-flare category | TBD | TBD |
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | Rare class handling required |
| 4 | TBD | Highest retained severity | TBD | Rare class handling required |

The contract must explicitly address:

- whether A-class is retained;
- whether no-flare samples form a separate class or are mapped elsewhere;
- how multiple events are reduced to one target;
- how labels from different source catalogs are reconciled;
- whether supplied labels and flux-derived bins are exactly consistent.

## 6. Bin contract

Canonical bins must satisfy

\[
-\infty=b_0<b_1<\cdots<b_K=+\infty,
\qquad B_k=[b_k,b_{k+1}).
\]

Threshold equality belongs to the bin on the right. All thresholds must be stored in the same numeric coordinate as \(Z\).

| Class index | Flare class | Lower threshold | Upper threshold | Coordinate |
|---:|---|---:|---:|---|
| 0 | TBD | \(-\infty\) or TBD | TBD | raw/transformed flux TBD |
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | \(+\infty\) or TBD | TBD |

The exact A/B/C/M/X thresholds are intentionally left `TBD` because the supplied OCQR documents require a dataset-specific validation artifact but do not provide numerical values or a finalized taxonomy.

## 7. Required consistency validation

Every retained observation must satisfy at least

\[
Y_{\mathrm{ord}}=k\Longrightarrow Z\in B_k.
\]

For the canonical disjoint and exhaustive bin contract, the preferred validated relation is

\[
Y_{\mathrm{ord}}=k\Longleftrightarrow Z\in B_k.
\]

The validation artifact must report:

- total retained samples;
- inconsistency count and rate;
- inconsistency counts by class and source period;
- missing or nonfinite target count;
- boundary-equality cases;
- samples excluded or corrected and the exact rule;
- hashes of input tables and retained manifests.

Any unresolved retained inconsistency places those observations outside the theorem-backed canonical target unless the target is redefined in a separate variant.

## 8. Chronological split policy

The project theory identifies a development period of 2010–2019 and a future holdout of 2020–2024. This design reduces direct future-test leakage but does not prove exchangeability.

A complete split record must specify:

| Partition | Intended period | Exact rule | Manifest/hash |
|---|---|---|---|
| Training | Within 2010–2019 | TBD | TBD |
| Validation | Within 2010–2019 | TBD | TBD |
| Calibration | Within 2010–2019 | TBD | TBD |
| Future test | 2020–2024 | Chronological holdout; exact endpoints TBD | TBD |

The exact date boundaries, gap policy, active-region grouping, and sample-window overlap rules must be recorded. Calibration data cannot be used to choose model checkpoints, bins, transformations, or variants.

## 9. Dependence and exchangeability caveat

Chronological separation is an empirical temporal-extrapolation design, not evidence that within-class calibration and test scores are exchangeable.

The experiment must separately analyze:

- repeated active-region dependence;
- overlapping observation windows;
- solar-cycle and temporal distribution shift;
- changes in instruments, catalogs, or preprocessing;
- class-conditional sample scarcity, especially for M and X classes.

Results must distinguish theorem-conditional coverage claims from empirical future-holdout performance.

## 10. Input modalities and preprocessing

| Component | Value |
|---|---|
| Primary active-region features/images | TBD |
| Additional modalities | TBD |
| Temporal alignment | TBD |
| Missing modality handling | TBD |
| Normalization | TBD |
| Training-only augmentation | TBD |
| Feature leakage checks | TBD |
| Forecast-window leakage checks | TBD |

Every input field must be available at the declared forecast issue time. Post-event or future-derived features are prohibited.

## 11. Automated validation requirements

Tests and validation scripts must verify:

1. finite numeric targets and model endpoints;
2. deterministic target transformation;
3. fixed supplied-label mapping to consecutive ordinal indices;
4. exact target-label-bin consistency for all retained samples;
5. right-bin equality at every internal threshold;
6. true-label Mondrian grouping;
7. no sample, active-region-window, or forbidden temporal overlap according to the declared policy;
8. chronological ordering of development and future test partitions;
9. per-class counts for every split;
10. explicit reporting of zero or insufficient calibration counts and resulting \(+\infty\) corrections;
11. hashes for source tables, retained manifests, configuration, checkpoints, and dataset contract;
12. raw-empty, fragmentation, hull-inflation, fallback-inflation, and final full-set metrics.

## 12. Required experiment metadata

- dataset release/catalog identifiers;
- observation and target source versions;
- target transformation parameters;
- flare taxonomy and threshold array;
- forecast horizon and input window;
- no-flare and multiple-event policy;
- split date boundaries;
- active-region grouping policy;
- split manifests and hashes;
- preprocessing configuration hash;
- code commit and random seed;
- checkpoint identifier;
- dataset-card version.

## 13. Known limitations

- The future holdout is subject to temporal distribution shift.
- Repeated active regions and overlapping windows can violate simple independence assumptions.
- Rare severe classes may yield infinite class-specific conformal corrections and inefficient prediction sets.
- The exact transformation, taxonomy, thresholds, no-flare policy, and dataset source are not fixed in the supplied documents.

## 14. Completion checklist

- [ ] Freeze prediction unit, forecast horizon, and observation window.
- [ ] Define source catalog and numeric target field.
- [ ] Freeze target transformation.
- [ ] Finalize class taxonomy and no-flare handling.
- [ ] Materialize threshold array in the target coordinate.
- [ ] Produce target-label consistency validation artifact.
- [ ] Freeze chronological and active-region-aware split manifests.
- [ ] Document modalities and leakage controls.
- [ ] Record license, release, and source URL.
- [ ] Add dependence and temporal-shift analyses.
