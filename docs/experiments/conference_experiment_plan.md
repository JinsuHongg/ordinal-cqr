# Conference-track OCQR experiment plan

## Frozen primary design

The initial conference suite contains RetinaMNIST and UTKFace. Both run at
`alpha=0.10` with seeds `[0,1,2,3,4]` once each split manifest is frozen.
RetinaMNIST uses the documented class-index embedding `Z=Y_ord`; UTKFace uses
chronological age and bins `[20,40,60,80]`. Every run has separate training,
validation, calibration, and test subsets. Checkpoints are selected before
calibration: total validation pinball loss for QR and validation cross-entropy
for classifiers.

The primary methods are LAC, APS, OAPS (only after common-evaluator validation),
and canonical OCQR v0.3. COPOC is excluded because no verified unimodal
backbone/protocol is recorded. The representative ablation dataset is
RetinaMNIST: OCQR-Pooled, OCQR-NoHull, OCQR-NoFallback, and OCQR. The latter two
are noncanonical analysis variants and must not receive the canonical guarantee.

Solar flare is explicitly **blocked**, not omitted silently. Its future test is
chronological extrapolation, with no intended direct test leakage, but is not an
exchangeable evaluation. It enters only after the documented portable path,
manifest/overlap audit, and validation-selected QR checkpoint are available.

## Required output contract

Each completed run is written under:

```text
outputs/conference_v0_3/<dataset>/<method>/seed_<seed>/
  config.yaml  provenance.json  calibration.json  metrics.json  predictions.*
```

`provenance.json` records method and dataset-card versions, seed, split ID and
hash, resolved configuration hash, commit, checkpoint, criteria, timestamp,
runtime, and hardware. `calibration.json` preserves infinity as the explicit
string `"+inf"`, never a surrogate finite value. Predictions contain the true
ordinal label, raw/final masks, and numeric target so every reported metric can
be recomputed without retraining.

The aggregate command validates those required fields and creates only
machine-generated artifacts in `results/conference_v0_3/`. Runs failing schema
or version checks are rejected rather than mixed into a manuscript table.

## Reporting

Primary metrics are marginal, macro-class, and worst-class coverage; mean and
median cardinality; mean ordinal span; singleton, full-set, contiguous, and
fragmented rates. OCQR additionally reports raw coverage/size/empty rate,
fragmentation, hull/fallback/total inflation, and calibration diagnostics.
Per-class tables include class test counts. Repeated scalar results are reported
as mean ± standard deviation across end-to-end seeds.
