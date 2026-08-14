---
dataset_id: solar_flare
name: Solar Flare Prediction Dataset
card_version: "0.3.0"
status: provisional
project: ordinal-conformal-prediction
method_compatibility:
  ocqr: "0.3.0"
representation_type: observed_numeric_target
input_modality:
  - 13-channel SuryaBench SDO image stack (HMI and AIA channels)
numeric_target:
  symbol: Z
  source: peak_xray_flux
  transformation: "log10(max_intensity) + 9"
  unit: "log10(W m^-2) + 9"
ordinal_label:
  symbol: Y_ord
  source: max_goes_class
class_count: 5
bins:
  convention: "B_k = [b_k, b_{k+1})"
  threshold_equality: right_bin
  thresholds: [2.0, 3.0, 4.0, 5.0]
split_policy: chronological
license: "CC BY 4.0"
source_url: "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting"
last_updated: "2026-08-13"
---

# Solar Flare Prediction Dataset Card and OCQR Metadata

## 1. Purpose

The solar-flare dataset is a temporal-shift benchmark for OCQR. It evaluates ordinal uncertainty quantification under severe class imbalance and chronological distribution shift.

The numeric target is the maximum GOES X-ray flux in the 24-hour prediction window, transformed deterministically for quantile regression. The supplied maximum flare class is the ordinal label used for Mondrian calibration. `FQ` is an event-label statement, not a claim that the maximum background flux is below the B-class threshold.

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
| Prediction unit | One row at one `timestamp` in the flare-index CSV, retained only when the configured image times exist |
| Forecast horizon | 24 hours: the prediction window is \([t,t+24\mathrm{h})\) for image time \(t\) |
| Observation window | One configured 13-channel SuryaBench image stack at offset `[0]` minutes in the reviewed conference configurations |
| Target event selection | `max_goes_class` is the maximum GOES flare class in the 24-hour prediction window; `max_intensity` is the maximum GOES X-ray flux in that same window |
| Multiple events in horizon | Upstream maximum over the 24-hour window; not recomputed by the adapter |
| No-flare handling | `FQ` means no flare in the 24-hour prediction window, but `max_intensity` still records the maximum background GOES X-ray flux |
| Active-region identifier | Not present in the reviewed split CSV schema |
| Timestamp convention | Naive CSV timestamp parsed by pandas; timezone is not stated in the adapter |

These decisions affect dependence, labels, sample counts, and reproducibility and must be frozen before calibration.

## 4. Numeric target

| Field | Value |
|---|---|
| Source quantity | Peak X-ray flux |
| Source instrument/catalog | SuryaBench flare-forecasting CSV; GOES class is provided in `max_goes_class` |
| Native unit | W m^-2 for GOES X-ray flux, as implied by the standard GOES class thresholds |
| Canonical transformation | `log10(max_intensity) + 9` |
| Nonpositive/missing values | `max_intensity == 0` raises; missing, negative, and nonfinite values require preflight rejection because the log transform is undefined or nonfinite |
| Finite-value requirement | Required |
| Precision and dtype | Floating point |

If a logarithmic or other deterministic transformation is used, all thresholds must be represented in the transformed coordinate. The transformation cannot be selected using calibration or test outcomes.

The reviewed split CSVs contain the observed maximum GOES X-ray flux, including background flux for `FQ` windows. The legacy utility `src/ordinal_cqr/utils/index_file_creator.py` instead synthesizes `max_intensity` from `max_goes_class` and assigns `FQ` a fixed `1e-9`; it must not be used to regenerate the canonical raw-flux manifests described here.

## 5. Ordinal label taxonomy

The current adapter maps the supplied `max_goes_class` values into five ordinal classes.

| Canonical index | Supplied class | Meaning | Retained? | Notes |
|---:|---|---|---|---|
| 0 | FQ or A | No-flare/quiet (`FQ`) or A-class flare | Yes, after the retained-population rule | Both map to 0 in `_map_goes_class` |
| 1 | B | B-class flare | Yes | |
| 2 | C | C-class flare | Yes | |
| 3 | M | M-class flare | Yes | Rare class handling required |
| 4 | X | X-class flare | Yes | Rare class handling required |

`FQ` and A-class values are intentionally merged into class 0. The upstream benchmark, rather than this adapter, determines event aggregation. A retained-row validation artifact must still establish consistency between supplied classes and transformed flux.

## 6. Bin contract

Canonical bins must satisfy

\[
-\infty=b_0<b_1<\cdots<b_K=+\infty,
\qquad B_k=[b_k,b_{k+1}).
\]

Threshold equality belongs to the bin on the right. All thresholds must be stored in the same numeric coordinate as \(Z\).

| Class index | Flare class | Lower threshold | Upper threshold | Coordinate |
|---:|---|---:|---:|---|
| 0 | FQ/A | \(-\infty\) | 2 | `log10(W m^-2) + 9` |
| 1 | B | 2 | 3 | `log10(W m^-2) + 9` |
| 2 | C | 3 | 4 | `log10(W m^-2) + 9` |
| 3 | M | 4 | 5 | `log10(W m^-2) + 9` |
| 4 | X | 5 | \(+\infty\) | `log10(W m^-2) + 9` |

The thresholds are the standard GOES decade boundaries expressed after the configured log transform. The raw files contain high-background `FQ` rows and 12 test `M0.9` rows with flux \(9\times10^{-6}\), below the M threshold. The retained-population rule below removes those inconsistencies before model training, calibration, and evaluation.

## 7. Required consistency validation

Every retained observation must satisfy at least

\[
Y_{\mathrm{ord}}=k\Longrightarrow Z\in B_k.
\]

For the canonical disjoint and exhaustive bin contract, the preferred validated relation is

\[
Y_{\mathrm{ord}}=k\Longleftrightarrow Z\in B_k.
\]

The raw files demonstrate that the unfiltered \(Z\)/\(Y_{\mathrm{ord}}\) pair fails the forward consistency condition for high-background FQ rows:

| Split | Class-0 rows | Rows with \(Z\geq2\) |
|---|---:|---:|
| Training | 20,130 | 4,304 |
| Validation | 761 | 192 |
| Calibration (`leaky_validation.csv`) | 1,443 | 249 |
| Test | 8,346 | 468 |

The canonical raw-flux variant applies this fixed rule before image-availability filtering in every partition:

1. normalize `max_goes_class` by stripping whitespace and uppercasing;
2. exclude `FQ` only when `max_intensity >= 1e-7` (the B-class boundary, inclusively);
3. exclude the exact malformed label `M0.9`.

This retains low-background FQ rows and all other supported labels. The resulting source-manifest sizes are 70,456 training, 3,480 validation, 5,799 calibration, and 43,368 test rows, before filtering rows without an available image. Under the reviewed files, the retained rows satisfy the stated target-label-bin contract. Because this is a target-dependent population restriction, it must be declared in every experiment and the coverage claim applies only to this retained population.

For the selected variant, the validation artifact must report:

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
| Training | 2010-05-13 00:00 through 2019-12-31 23:00 | `train.csv`, 74,760 raw / 70,456 retained rows | SHA-256 `2ec7b8f39367f8340a39889bc66525aff303410d7b7ce6c12a55ea346b55e865` |
| Validation | 2011-01-15 00:00 through 2019-01-31 23:00 | `validation.csv`, 3,672 raw / 3,480 retained rows | SHA-256 `803d2e5584fe9bbe23bc02cbed1b06fb47520e4863c2b22b5f09f9d5c654c658` |
| Calibration | 2011-01-01 00:00 through 2019-02-14 23:00 | `leaky_validation.csv`, 6,048 raw / 5,799 retained rows | SHA-256 `03134a82a53891d25761774c5aad52f77e01673195f7cfd28c0dc061bfe5849e` |
| Future test | 2020-01-01 00:00 through 2024-12-31 23:00 | `test.csv`, 43,848 raw / 43,368 retained rows | SHA-256 `40ddef01aebe23e5ee460717a08b7392827eacca2852af074d5f1533f59ebd4b` |

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
| Primary features/images | All 13 channels in the configured 224×224 stacked SuryaBench Zarr array, in stored channel order |
| Additional modalities | HMI and AIA channels as declared by the Zarr `channel_names` metadata; the preflight and statistics artifacts must preserve their exact order |
| Temporal alignment | Exact timestamp lookup; required offsets must all exist in the Zarr timestamp index |
| Missing modality handling | Exclude timestamps lacking a required image or matching flare-index row |
| Normalization | Signed `log1p` pixel transform, then standardization using configured dataset mean and standard deviation |
| Training-only augmentation | None in `FlareSuryaBenchDataset` |
| Feature leakage checks | Exact pairwise timestamp intersections are rejected and source CSV hashes are verified; active-region grouping cannot be checked because the reviewed CSV schema has no active-region identifier |
| Forecast-window leakage checks | Cross-split timestamps within the 24-hour forecast horizon are counted in `split_audit.json`; proximity is reported as dependence rather than silently treated as direct overlap |

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
- The target-dependent retained-population rule changes the estimand: results do not describe high-background no-flare windows or malformed `M0.9` records.
- The reviewed repository configuration points to `/scratch/users/jhong36/data`, while the inspected mounted assets are under `/mnt/storage/surya`; this path must be made portable without tracking machine-specific paths.

## 14. Completion checklist

- [x] Record the 24-hour forecast horizon and target-window semantics.
- [x] Define source catalog and numeric target field.
- [x] Freeze the current target transformation.
- [x] Finalize the current class taxonomy and no-flare handling.
- [x] Materialize the threshold array in the target coordinate.
- [x] Freeze the retained-population rule for label-consistent raw-flux targets.
- [ ] Produce the target-label consistency validation artifact for that selected variant.
- [ ] Freeze chronological and active-region-aware split manifests.
- [ ] Document modalities and leakage controls.
- [ ] Record license, release, and source URL.
- [ ] Add dependence and temporal-shift analyses.
