# Conference-track OCQR experiment audit

**Audit date:** 2026-08-05
**Canonical method:** `ocqr` v0.3.0
**Status:** pre-implementation audit; no core-method changes preceded this record.

## Authoritative references reviewed

- `docs/methods/ocqr_theory.md` and `docs/methods/ocqr_contract.md` (v0.3.0).
- `docs/manuscript_scope.md`.
- Dataset cards in `docs/datasets/` for RetinaMNIST, UTKFace, solar flare,
  EyePACS, and Adience.
- Existing training, calibration, data-module, metric, test, configuration,
  checkpoint, and result artifacts.

## Repository inventory

| Area | Current location | Audit finding |
|---|---|---|
| Datasets | `src/ordinal_cqr/datasets/` | Adience, UTKFace, EyePACS, solar flare adapters; RetinaMNIST adapter lives with its data module. |
| Data modules | `src/ordinal_cqr/datamodules/` | Four-way logical split exists for UTKFace/EyePACS/Adience; RetinaMNIST instead has train/calibration plus the upstream validation/test splits. No module persists manifests or split hashes. |
| Models/training | `src/ordinal_cqr/models/`, `scripts/experiments/training.py` | ResNet classification and quantile-regression heads; QR configurations use 0.05/0.50/0.95 quantiles. |
| OCQR calibration/prediction | `src/ordinal_cqr/explainability/poshoc_uc.py`, `scripts/experiments/calibration.py` | `OrdinalCQRWrapper` implements the central v0.3 path. The calibration script saves a partial strict-JSON provenance payload. |
| Classification baselines | `poshoc_uc.py`, `scripts/experiments/cls_calibration.py`, `scripts/experiments/ordinal_calibration.py` | LAC (`ClsCPWrapper`), APS, OAPS, min-CPS, min-RCPS, COPOC, and risk-control wrappers exist, but there is no unified canonical experiment driver or common metric schema. |
| Metrics | `src/ordinal_cqr/metrics/classification_metrics.py` | OCQR raw/final diagnostics are implemented. General classification metrics omit required per-class coverage, spans, singleton/full/fragmentation rates, and components. |
| Configurations | `configs/qr/`, `configs/cls/`, `configs/method/ocqr.yaml` | v0.3 method YAML exists. Per-dataset configs are legacy-style and write shared asset paths; solar QR training incorrectly configures MSE instead of pinball loss. |
| Tests | `tests/` | Focused OCQR mechanics, batch interface, post-processing metrics, metadata, and EyePACS tuning tests exist. No synthetic repeated coverage validation or conference pipeline tests. |
| Previous outputs | `assets/uc_results/`, `docs/results/`, root `all_datasets_mondrian_results.csv`, `results/utkface_benchmark_results.md` | All are legacy/incompatible for canonical tables: no canonical run directories, split/config hashes for every row, method-version linkage, full diagnostics, or evidence that calibration/test were untouched during selection. |
| Checkpoints | `assets/checkpoints/{qr,cls}` | RetinaMNIST, UTKFace, EyePACS, and Adience checkpoints exist. No solar QR checkpoint is present. Filenames expose a validation loss but do not establish a frozen split manifest or selection provenance. |
| Publication generation | none | No CSV/JSON aggregation pipeline or table/figure generator was found. |

## Contract-compliance review

| Requirement | Expected behavior | Current implementation | Status | Required action |
|---|---|---|---|---|
| Separate `Z` and `Y_ord` | Canonical three-field batches | OCQR wrapper requires `(X,Z,Y_ord)` except an explicit compatibility flag; reviewed adapters emit both | Pass | Keep derived-label compatibility disabled in canonical configs. |
| Target-label-bin consistency | Validate supplied label against declared bins | `_validate_targets` checks bucketized `Z` against `Y_ord` | Pass | Add dataset-level validation artifacts to runs. |
| Floating bin comparisons / right boundary | Promote dtype; `[b_k,b_{k+1})`, equality right | `_class_indices` promotes dtype and uses `bucketize(..., right=True)` | Pass | Retain deterministic tests; add public pipeline coverage. |
| Quantile crossing | Sort endpoints before all downstream use | `_ordered_finite_endpoints` uses `minimum`/`maximum` | Pass | None. |
| Nonfinite endpoints / targets | Raise explicitly | Async finite assertions in endpoint and target validation | Pass | Add `inf` endpoint unit test (only NaN is currently explicit). |
| Negative scores/corrections | Keep valid values | CQR score is unclipped; finite negative `q` is supported | Pass | Add focused test for finite negative correction score inversion. |
| Exact augmented order statistic / ties | `ceil((N+1)(1-alpha))`; implicit appended `+inf`; no interpolation | `kthvalue` only after rank attainability; otherwise `+inf`; tie count retained | Pass | Clarify metadata terminology (`q_hat` -> canonical `q_k`) in new outputs. |
| `N_k=0`, unattainable rank, `q_k=+inf` | Conservative infinite candidate correction | Implemented and tested | Pass | Surface in diagnostics and experiments. |
| True-label grouping | Group calibration by supplied `Y_ord` | Uses validated supplied `y_ord` | Pass | None. |
| Candidate-specific corrections | Test every candidate with its own `q_k` | Vectorized candidate intervals use `q_hats[None, :]` | Pass | Add regression test that distinguishes candidate corrections from point-class correction. |
| Score inversion / empty numeric intervals | Exact inversion; retain empty finite interval | Candidate `L/U` overlap checks use `candidate_lo <= candidate_hi` | Pass | Add explicit inversion and empty-interval tests. |
| Raw set, fallback, hull | Raw candidates, pre-hull full fallback, add-only ordinal hull | `_ordinal_hull` falls back to all labels and fills gaps | Pass | Expose NoHull and NoFallback only as named analysis variants. |
| Raw/final metrics | Required diagnostics plus common conference metrics | OCQR metric includes raw coverage/size/empty/fragmented/inflations; lacks median, span, singleton, per-class raw diagnostics | Partial | Build shared metric/report schema. |
| Calibration metadata | Counts/ranks/corrections/ties/min/max and provenance | OCQR payload covers core calibration metadata; strict JSON represents infinity as `q_hat: null` plus boolean | Partial | Add explicit string representation (`"+inf"`), dataset-card version, training criterion, hardware/runtime, test counts, and run-local paths. |
| Split/config hashes | Stable manifests and hashes per run | Solar source-index hashes exist; generic modules report unavailable stable manifests | Partial | Materialize deterministic manifests and hashes before canonical experiments. |
| Method-version synchronization | v0.3 everywhere | Method config and OCQR JSON use v0.3.0; legacy configs/results lack consistent linkage | Partial | New configs/results must carry v0.3.0; quarantine legacy outputs. |

## Dataset readiness and selection

| Dataset | Intended role | Contract readiness | Split/readiness finding | Decision |
|---|---|---|---|---|
| RetinaMNIST | Class-only ordinal image dataset | Documented; `Z=Y_ord`, five midpoint bins | Upstream train/val/test plus a deterministic 70/30 train/cal split, but no persisted manifest and not a four-way 60/10/20/10 split | **Primary after manifest work**. Lowest-cost focused image benchmark. |
| UTKFace | Numeric-age ordinal image dataset | Documented numeric target and bins | Four-way deterministic row-level split exists but files/manifests/hashes are not persisted | **Primary after manifest work**. Provides the measured-target role. |
| Solar flare | Severely imbalanced temporal-extrapolation application | Detailed provisional card, label/target filtering policy and source hashes | Missing local checkpoint; configured paths are machine-specific; chronology overlaps development partitions and exchangeability/active-region audits remain unfinished | **Conference dataset, blocked from aggregation until blockers clear**. Treat as temporal extrapolation, never as proof of exchangeability. |
| EyePACS | Alternative class-only imbalanced medical dataset | Provisional, documented class-index embedding | Image-level split has no manifests or patient/eye leakage control | Supplementary candidate only; not selected ahead of RetinaMNIST. |
| Adience | Alternative ordinal image dataset | Provisional representative-age target | Subject recurrence and manifests unresolved | Exclude from focused suite. |

The frozen conference suite is **RetinaMNIST, UTKFace, and solar flare**. Solar
results can enter only after a portable data configuration, frozen
chronology-aware manifest, overlap audit, and QR checkpoint selected solely on
validation pinball loss are available. The solar evaluation must state: no
direct future test leakage is intended, it is chronological extrapolation, and
chronology does not establish exchangeability.

## Existing-result audit

The following artifacts must be marked **legacy** and excluded from
`results/conference_v0_3/`: `assets/uc_results/**`, `docs/results/*.csv`,
`all_datasets_mondrian_results.csv`, and `results/utkface_benchmark_results.md`.
They do not provide the required jointly linked method version, dataset-contract
version, run-local config/provenance, split manifest hash, selection protocol,
sample-level predictions, or current metric schema. The checkpoint inventory is
reusable only after its upstream training config and validation-only selection
criterion are recorded in new provenance; it is not itself a canonical result.

## Reusable components

- Vectorized canonical OCQR calibration/prediction, crossing correction, and
  nonempty contiguous post-processing.
- Target-label-bin validation and solar retained-population audit helpers.
- QR and classification ResNet18 training paths and saved checkpoints.
- Partial OCQR calibration/evaluation strict-JSON serializers.
- Existing LAC, APS, and OAPS wrappers, subject to unified evaluation.

## Missing components and blockers

1. A canonical run runner that freezes splits/checkpoints, saves predictions,
   provenance, calibration, and metrics in unique run directories.
2. Persisted deterministic split manifests and hashes for RetinaMNIST/UTKFace.
3. Common baseline evaluator and metric schema implementing all conference
   aggregate, per-class, ordinal-span, fragmentation, and point metrics.
4. Explicit OCQR ablation controls (`OCQR-Pooled`, `OCQR-NoHull`, and
   analysis-only `OCQR-NoFallback`).
5. Result aggregation plus CSV/strict-JSON to LaTex/PDF/PNG generation.
6. Synthetic repeated-draw validation and the missing boundary/score tests.
7. Solar data/checkpoint/portable-path and chronology-overlap blockers above.

## Recommended minimal conference suite

- Datasets: RetinaMNIST, UTKFace, and solar flare. Solar remains blocked from
  aggregation until its temporal-split prerequisites are evidenced.
- Methods: OCQR, LAC, APS, OAPS, and COPOC. Every baseline requires common
  metric and split-discipline validation; COPOC additionally requires its
  verified unimodal backbone.
- Ablations: `OCQR-Pooled`, `OCQR-NoHull`, `OCQR-NoFallback` on RetinaMNIST.
- Protocol: alpha 0.10; seeds `[0,1,2,3,4]` only after end-to-end automation
  exists. Initial validation should use seed 0 and identify it as such.
- Selection: QR checkpoint by validation total pinball loss; classification
  checkpoint by validation cross-entropy/NLL. Calibration and test data are
  never used for selection.

## Audit conclusion

Core OCQR v0.3 mechanics are sufficiently close to the normative contract to
build the conference layer without changing the method. The next phase should
add isolated experiment infrastructure, metrics, tests, and documentation;
legacy outputs must remain quarantined. No real-data canonical claim can be
made until a run has a frozen split, provenance, saved predictions, and passed
validation.
