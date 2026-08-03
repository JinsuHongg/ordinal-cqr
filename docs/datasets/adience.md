---
dataset_id: adience
name: Adience
card_version: "0.1.0"
status: provisional
project: ordinal-conformal-prediction
method_compatibility:
  ocqr: "0.3.0"
representation_type: undecided_ordinal_age_group
input_modality:
  - face_image
numeric_target:
  symbol: Z
  source: TBD
  transformation: TBD
ordinal_label:
  symbol: Y_ord
  source: supplied_ordered_age_group
class_count: TBD
bins:
  convention: "B_k = [b_k, b_{k+1})"
  threshold_equality: right_bin
  thresholds: TBD
split_policy: TBD
license: TBD
source_url: TBD
last_updated: "2026-08-03"
---

# Adience Dataset Card and OCQR Metadata

## 1. Purpose

Adience is included as an ordinal age-group image dataset. The supplied project documents identify it as part of the evaluation scope but do not define its canonical numeric target \(Z\). Therefore, this card is provisional until one representation is selected and frozen.

Canonical experiments must not begin until the target representation, bins, and consistency rule are completed.

## 2. Canonical OCQR interface

Each retained observation must expose

\[
(X,Z,Y_{\mathrm{ord}}),
\]

where:

- \(X\): face image;
- \(Y_{\mathrm{ord}}\): supplied ordered age-group label mapped to consecutive indices;
- \(Z\): numeric target selected by the finalized representation contract.

The two admissible high-level choices currently identified are:

1. a fixed class-only embedding, such as \(Z=Y_{\mathrm{ord}}\);
2. another documented numeric age coordinate supported by available metadata.

The supplied documents do not determine which choice is canonical.

## 3. Representation decision

### Option A: class-index embedding

\[
Z=Y_{\mathrm{ord}}.
\]

Under this option, Adience is treated as a class-only ordinal dataset. Midpoint thresholds between adjacent embedded classes define the bins. The coordinate is surrogate rather than measured age.

### Option B: observed or derived numeric age coordinate

Under this option, the exact source field or deterministic derivation must be documented. Every retained sample must satisfy the declared target-label-bin consistency relation. Ambiguous interval labels or overlapping age groups must be resolved by a deterministic policy before calibration.

### Required canonical decision

| Field | Value |
|---|---|
| Selected option | TBD |
| Justification | TBD |
| Variant name | TBD |
| Numeric target definition | TBD |
| Target transformation | TBD |

Alternative representations must be reported as separate method variants or ablations.

## 4. Label mapping

| Canonical index | Source age-group label | Ordered meaning | Retained? |
|---:|---|---|---|
| 0 | TBD | Youngest retained group | TBD |
| 1 | TBD | TBD | TBD |
| ... | ... | ... | ... |
| \(K-1\) | TBD | Oldest retained group | TBD |

The exact source labels, handling of uncertain labels, class count, and mapping are not specified in the supplied OCQR files.

## 5. Bin contract

All canonical bins must satisfy

\[
-\infty=b_0<b_1<\cdots<b_K=+\infty,
\qquad B_k=[b_k,b_{k+1}).
\]

Equality at an internal threshold belongs to the bin on the right.

### If Option A is selected

Use the class-index embedding with midpoint thresholds. The final threshold array depends on the confirmed number of retained classes.

### If Option B is selected

Define thresholds in the same numeric coordinate as \(Z\). If source age groups overlap, are open-ended, or are ambiguous, the canonical retained subset and deterministic resolution rule must be explicitly stated. A representation that cannot establish the forward consistency implication is outside the canonical theorem claim.

## 6. Required consistency condition

For canonical disjoint and exhaustive bins, every retained observation should satisfy

\[
Y_{\mathrm{ord}}=k\Longleftrightarrow Z\in B_k.
\]

At minimum,

\[
Y_{\mathrm{ord}}=k\Longrightarrow Z\in B_k
\]

must be established for the theorem to apply.

## 7. Preprocessing and exclusions

| Component | Value |
|---|---|
| Image source/crop | TBD |
| Face detection or alignment | TBD |
| Resize and normalization | TBD |
| Training augmentation | TBD |
| Ambiguous label handling | TBD |
| Unknown/invalid label handling | TBD |
| Duplicate or near-duplicate handling | TBD |
| Subject identity handling | TBD |

All rules must be fixed before calibration data are accessed.

## 8. Split policy and fold provenance

Adience may be distributed with predefined folds or subject-related structure, but the supplied project documents do not specify which artifacts will be used. The canonical experiment must record the exact fold or manifest policy rather than relying only on a generic dataset name.

| Field | Value |
|---|---|
| Original fold source | TBD |
| Split unit | TBD |
| Train manifest/hash | TBD |
| Validation manifest/hash | TBD |
| Calibration manifest/hash | TBD |
| Test manifest/hash | TBD |
| Split seed | TBD |

Any subject recurrence across splits must be measured and reported.

## 9. Automated validation requirements

Tests must verify:

1. all retained source labels map to consecutive ordered indices;
2. the canonical representation option is recorded in configuration;
3. \(Z\) and \(Y_{\mathrm{ord}}\) are logically distinct batch fields;
4. every retained sample satisfies the declared target-label-bin relation;
5. threshold equality follows the right-bin convention;
6. integer and floating-point bin comparisons are safe;
7. ambiguous or invalid labels follow one frozen rule;
8. split manifests do not overlap under the declared split unit;
9. per-class counts and exclusions are reported;
10. dataset-contract, manifest, and configuration hashes are persisted.

## 10. Known limitations

- The canonical numeric target is unresolved.
- Different ordinal embeddings can change efficiency and candidate membership.
- Ambiguous or interval-valued source labels may complicate target-label consistency.
- Subject and image dependence may weaken an exchangeability interpretation.

## 11. Completion checklist

- [ ] Select and name the canonical numeric representation.
- [ ] Confirm source age-group labels and ordering.
- [ ] Define class count and exact thresholds.
- [ ] Freeze ambiguous/invalid label policy.
- [ ] Document preprocessing and split/fold use.
- [ ] Record license, release, and source URL.
- [ ] Generate stable manifests and hashes.
- [ ] Add representation-specific consistency tests.
