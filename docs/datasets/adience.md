---
dataset_id: adience
name: Adience
card_version: "0.2.0"
status: provisional
project: ordinal-conformal-prediction
method_compatibility:
  ocqr: "0.3.0"
representation_type: fixed_age_group_representative
input_modality:
  - face_image
numeric_target:
  symbol: Z
  source: fixed_age_group_representative
  transformation: identity
ordinal_label:
  symbol: Y_ord
  source: supplied_ordered_age_group
class_count: 8
bins:
  convention: "B_k = [b_k, b_{k+1})"
  threshold_equality: right_bin
  thresholds: [3.0, 7.75, 14.0, 23.0, 34.5, 45.5, 57.75]
split_policy: "pooled five fold files; stratified row-level 60/10/20/10 split; seed 42"
license: "not established from the reviewed local copy; verify before redistribution"
source_url: "https://talhassner.github.io/home/projects/Adience/Adience-data.html"
last_updated: "2026-08-03"
---

# Adience Dataset Card and OCQR Metadata

## 1. Purpose

Adience is included as an ordinal age-group image dataset. The current adapter uses a fixed representative age for each source age group as \(Z\). This makes the implementation reviewable, but the representative-age convention and row-level split still require a manuscript-level justification and frozen manifests.

Canonical reporting still requires frozen retained-sample manifests and a full target-label-bin validation artifact.

## 2. Canonical OCQR interface

Each retained observation must expose

\[
(X,Z,Y_{\mathrm{ord}}),
\]

where:

- \(X\): face image;
- \(Y_{\mathrm{ord}}\): supplied ordered age-group label mapped to consecutive indices;
- \(Z\): fixed representative age for the supplied age group.

The current implementation uses fixed group representatives. It is not an exact observed age: the source labels are intervals, not individual ages.

## 3. Target representation

| Field | Value |
|---|---|
| Selected option | Fixed age-group representative |
| Justification | Deterministic scalar coordinate compatible with the nonoverlapping retained age groups |
| Variant name | `adience_age_group_representative_v1` |
| Numeric target definition | `(0,2)→1`, `(4,6)→5`, `(8,12)→10`, `(15,20)→17.5`, `(25,32)→28.5`, `(38,43)→40.5`, `(48,53)→50.5`, `(60,100)→65` years |
| Target transformation | Identity |

Alternative embeddings, including \(Z=Y_{\mathrm{ord}}\), must be reported as separate method variants or ablations.

## 4. Label mapping

| Canonical index | Source age-group label | Ordered meaning | Retained? |
|---:|---|---|---|
| 0 | `(0, 2)` | 1 year representative | Yes |
| 1 | `(4, 6)` | 5 year representative | Yes |
| 2 | `(8, 12)` | 10 year representative | Yes |
| 3 | `(15, 20)` | 17.5 year representative | Yes |
| 4 | `(25, 32)` | 28.5 year representative | Yes |
| 5 | `(38, 43)` | 40.5 year representative | Yes |
| 6 | `(48, 53)` | 50.5 year representative | Yes |
| 7 | `(60, 100)` | 65 year representative | Yes |

The current adapter retains only these eight exact source labels. Other source age strings are excluded; no interval-overlap resolution is performed.

## 5. Bin contract

All canonical bins must satisfy

\[
-\infty=b_0<b_1<\cdots<b_K=+\infty,
\qquad B_k=[b_k,b_{k+1}).
\]

Equality at an internal threshold belongs to the bin on the right.

The implemented representative coordinate uses thresholds `[3.0, 7.75, 14.0, 23.0, 34.5, 45.5, 57.75]`, the midpoints between adjacent representatives. The open-ended source group `(60, 100)` is represented by 65; this is a modeling convention rather than an observed age and must be reported as such.

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
| Image source/crop | `faces/{user_id}/coarse_tilt_aligned_face.{face_id}.{original_image}` |
| Face detection or alignment | Uses provider `coarse_tilt_aligned_face` files; no additional alignment in the adapter |
| Resize and normalization | Resize 224 × 224; ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Training augmentation | Random horizontal flip |
| Ambiguous label handling | Retain only the eight exact interval strings listed above |
| Unknown/invalid label handling | Drop rows missing age, user ID, face ID, or original image; drop noncanonical age labels and rows whose constructed image path is absent |
| Duplicate or near-duplicate handling | Not implemented |
| Subject identity handling | `user_id` is retained only to construct paths; it is not used to split or deduplicate |

All rules must be fixed before calibration data are accessed.

## 8. Split policy and fold provenance

The source distribution supplies five fold files, but the current adapter pools them and creates a new split. The canonical experiment must record the exact manifest policy rather than relying only on a generic dataset name.

| Field | Value |
|---|---|
| Original fold source | `fold_0_data.txt` through `fold_4_data.txt`, pooled before splitting |
| Split unit | Dataset row/image |
| Train manifest/hash | Not persisted; must be generated before canonical reporting |
| Validation manifest/hash | Not persisted; must be generated before canonical reporting |
| Calibration manifest/hash | Not persisted; must be generated before canonical reporting |
| Test manifest/hash | Not persisted; must be generated before canonical reporting |
| Split seed | 42 |

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

- The representative-age target is a surrogate; in particular, 65 is not the midpoint of `(60, 100)`.
- Different ordinal embeddings can change efficiency and candidate membership.
- Ambiguous or interval-valued source labels may complicate target-label consistency.
- Subject and image dependence may weaken an exchangeability interpretation.

## 11. Completion checklist

- [x] Select and name the current numeric representation.
- [x] Confirm source age-group labels and ordering.
- [x] Define class count and exact thresholds.
- [x] Freeze the current ambiguous/invalid label policy.
- [x] Document preprocessing and split/fold use.
- [ ] Verify provider license and release terms before redistribution.
- [ ] Generate stable manifests and hashes.
- [ ] Add representation-specific consistency tests.
