# Conference-track OCQR experiment plan

## Frozen primary design

The conference suite is limited to RetinaMNIST, UTKFace, and solar-flare
forecasting. All methods run at `alpha=0.10` with end-to-end seeds
`[0,1,2,3,4]`. The complete external baseline set is LAC, APS, OAPS, and COPOC;
the proposed method is canonical OCQR v0.3. No other datasets, external
baselines, miscoverage levels, sensitivity studies, embedding sweeps, or
calibration-size experiments belong in this conference suite.

| Dataset | Role and fixed target interface |
| --- | --- |
| RetinaMNIST | Class-only ordinal setting, with the fixed class-index embedding `Z=Y_ord`. |
| UTKFace | Observed numeric-target setting: chronological age is `Z`, with ordinal bins `[20,40,60,80]`. |
| Solar-flare forecasting | Observed numeric-target, highly imbalanced temporal-extrapolation setting, using the documented flare-intensity target and ordinal flare labels. |

EyePACS and Wine Quality are not part of this plan; they remain candidates for a
journal extension.

Every run has separate training, validation, calibration, and test subsets.
Checkpoint selection is completed before conformal calibration: quantile-
regression checkpoints use validation pinball loss, and classifier checkpoints
use validation cross-entropy. Calibration and test labels must not be used for
model selection, hyperparameter tuning, target/bin selection, or method
selection.

All four baselines use the same frozen split manifests and common evaluator
whenever technically possible. A baseline result enters manuscript aggregation
only after it passes common-evaluator validation and all required
provenance/schema checks.

### Conference question and primary comparison

The conference question is: **does candidate-specific class-conditional
calibration in OCQR improve per-class and worst-class coverage for ordinal
prediction while preserving useful prediction-set efficiency and ordinal
structure?** In particular, the main comparison asks whether OCQR improves the
reliability of the least-covered ordinal classes while retaining competitive set
size. Aggregation must make target coverage `(1-alpha)=0.90`, marginal
coverage, macro class-conditional coverage, worst-class coverage, and mean set
size directly comparable across methods.

## RetinaMNIST ablations

RetinaMNIST is the representative ablation dataset. The ablation rows are
OCQR-Pooled, OCQR-NoHull, OCQR-NoFallback, and OCQR.

- `OCQR-Pooled` isolates the contribution of class-specific Mondrian
  calibration by replacing it with pooled/global calibration.
- `OCQR-NoHull` is a diagnostic, noncanonical variant. It may retain the raw
  class-conditional inclusion property, but does not guarantee ordinal
  contiguity.
- `OCQR-NoFallback` is a diagnostic, noncanonical variant. It may retain the
  relevant inclusion property, but does not guarantee nonempty prediction sets.
- Canonical OCQR v0.3 combines class-conditional coverage, nonempty prediction
  sets, and ordinal contiguity under the stated method-contract assumptions.

`OCQR-NoHull` and `OCQR-NoFallback` must never be labeled canonical OCQR v0.3.

## Solar-flare temporal evaluation

Solar-flare forecasting uses a chronological future test split. Report its
results as temporal-extrapolation performance, not as empirical confirmation
that the exchangeability assumption holds. Before aggregation, each solar run
must record its frozen split manifest and hash, chronological split definition,
overlap/leakage audit, validation-selected checkpoint, train/validation/
calibration/test time ranges, and class counts in every split.

## Required output contract

Each completed run is written under:

```text
outputs/conference_v0_3/<dataset>/<method>/seed_<seed>/
  config.yaml  provenance.json  calibration.json  metrics.json  predictions.*
```

`provenance.json` includes at least the method version, dataset-card version,
seed, split ID and hash, resolved configuration hash, git commit, selected
checkpoint, checkpoint-selection criterion, timestamp, runtime, and hardware.
`calibration.json` preserves infinity as the explicit string `"+inf"`; it must
never replace infinity with an arbitrary large finite value. Predictions retain
the true ordinal label, numeric target, raw prediction mask/set, and final
prediction mask/set so all manuscript metrics can be recomputed without
retraining.

Aggregation rejects incompatible versions, missing required fields, invalid
schemas, and mismatched split/configuration hashes rather than silently mixing
runs.

## Reporting and computational overhead

Main conference tables report marginal coverage, macro class-conditional
coverage, worst-class coverage, mean prediction-set size, and full-set rate.
Per-class coverage tables include the number of test observations in each
class. Repeated scalar results are reported as mean ± standard deviation across
the five end-to-end seeds. A seed covers the full stochastic pipeline, not just
calibration. For a fixed official test set, retain that test set across seeds
and vary only allowed stochastic components and frozen training/calibration
construction.

Secondary machine-generated diagnostics, for ablations or supplementary
analysis, are median prediction-set size, mean ordinal span, singleton,
contiguous-set, and fragmented-set rates; raw coverage, raw set size, and raw
empty-set rate; hull, fallback, and total inflation; and class-specific
calibration counts `N_k`, conformal ranks `r_k`, corrections `q_k`, and the
number/rate of finite versus infinite `q_k`. They do not belong in the main
comparison table.

Each run also records calibration time, prediction/conformal post-processing
time, and total evaluation time; record samples per second where feasible.
Where measurable, separate base-model forward-pass time from conformal
post-processing time. Efficiency or scalability claims require these
measurements or a clear complexity argument. For OCQR, candidate-specific
post-hoc evaluation scales linearly with the number of ordinal classes `K`.
