---
dataset_id: utkface
name: UTKFace
card_version: "0.2.0"
status: provisional
project: ordinal-conformal-prediction
method_compatibility:
  ocqr: "0.3.0"
representation_type: observed_numeric_target
input_modality:
  - face_image
numeric_target:
  symbol: Z
  source: chronological_age
  transformation: identity
  unit: years
ordinal_label:
  symbol: Y_ord
  source: fixed_age_bin_index
class_count: 5
bins:
  convention: "B_k = [b_k, b_{k+1})"
  threshold_equality: right_bin
  thresholds: [20.0, 40.0, 60.0, 80.0]
split_policy: "order-dependent stratified image-level 60/10/20/10 split; seed 42; manifests required"
license: "non-commercial research use only (dataset provider statement)"
source_url: "https://susanqq.github.io/UTKFace/"
last_updated: "2026-08-03"
---

# UTKFace Dataset Card and OCQR Metadata

## 1. Purpose

UTKFace is used to evaluate OCQR when an observed numeric target is available. Chronological age serves as \(Z\), while a fixed age interval determines the ordinal class \(Y_{\mathrm{ord}}\).

The implemented canonical variant uses the exact age parsed from each filename and the fixed thresholds 20, 40, 60, and 80 years. Stable retained-sample manifests are not yet persisted, so this card remains provisional.

## 2. Canonical OCQR interface

Each retained observation must expose

\[
(X,Z,Y_{\mathrm{ord}}),
\]

where:

- \(X\): face image;
- \(Z\): observed chronological age, or a fixed deterministic transformation declared in this contract;
- \(Y_{\mathrm{ord}}\): index of the fixed age bin containing \(Z\).

Calibration grouping must use the supplied \(Y_{\mathrm{ord}}\). A derived-label interface is allowed only after this contract proves exact equivalence between the age field and the canonical bin assignment.

## 3. Numeric target

| Field | Canonical value |
|---|---|
| Source field | Chronological age from dataset metadata or filename parser |
| Unit | Years |
| Transformation | Identity |
| Valid range | 1--116 years in the reviewed local corpus (23,708 parseable `.jpg` files) |
| Missing/invalid handling | During split construction, filenames whose first underscore-delimited field cannot be parsed as a float are excluded. Image decode errors raise `RuntimeError`; no silent replacement is performed. |
| Floating-point representation | Required before threshold comparisons |

If a transformed age target is used, the transformation and all bin thresholds must be transformed consistently and the variant must be named separately.

## 4. Ordinal labels and bins

The canonical bins must form a disjoint ordered partition:

\[
-\infty=b_0<b_1<\cdots<b_K=+\infty,
\qquad B_k=[b_k,b_{k+1}).
\]

Equality at an internal threshold belongs to the bin on the right.

| Canonical class | Age interval | Lower threshold | Upper threshold | Source/justification |
|---:|---|---:|---:|---|
| 0 | \([ -\infty,20)\) years | \(-\infty\) | 20 | Filename age; configured threshold |
| 1 | \([20,40)\) years | 20 | 40 | Filename age; configured threshold |
| 2 | \([40,60)\) years | 40 | 60 | Filename age; configured threshold |
| 3 | \([60,80)\) years | 60 | 80 | Filename age; configured threshold |
| 4 | \([80,+\infty)\) years | 80 | \(+\infty\) | Filename age; configured threshold |

The exact thresholds must be frozen using training/validation design decisions only. They must not be selected using calibration or test performance.

## 5. Required consistency condition

For every retained canonical observation,

\[
Y_{\mathrm{ord}}=k\Longleftrightarrow Z\in B_k.
\]

At minimum, the forward implication required by the OCQR theorem must hold:

\[
Y_{\mathrm{ord}}=k\Longrightarrow Z\in B_k.
\]

Any sample violating the declared relation must be rejected, corrected by a documented deterministic rule, or placed outside canonical experiments.

## 6. Metadata extraction and validation

The implemented parser must document:

- the source of age metadata;
- filename parsing rules, if applicable;
- treatment of malformed filenames or records;
- whether age is integer-valued or may be fractional;
- duplicate-image handling;
- identity leakage controls, if identities can recur across splits;
- checksum or manifest generation.

The filename parser uses only the first underscore-delimited field. It does not use gender, race, or timestamp fields as model inputs.

## 7. Preprocessing

| Component | Value |
|---|---|
| Image decoding | Pillow `Image.open(...).convert("RGB")` |
| Face crop policy | Provider-distributed cropped/aligned image; no additional crop in the adapter |
| Resize | 128 × 128 |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Training augmentation | Random horizontal flip |
| Invalid image handling | Decode error raises; invalid filename is excluded before splitting |
| Demographic attributes used as inputs | None; gender and race filename fields are not read by the adapter |

Only \(X\), the declared numeric target, and the ordinal label may be passed through the canonical data interface. Any auxiliary attributes used for analysis must be separated from training inputs unless explicitly approved in configuration.

## 8. Split policy and dependence

| Field | Value |
|---|---|
| Split unit | Image filename |
| Identity-disjoint requirement | Not implemented; UTKFace filenames do not provide a subject identifier used by the adapter |
| Train manifest/hash | Not persisted; must be generated before canonical reporting |
| Validation manifest/hash | Not persisted; must be generated before canonical reporting |
| Calibration manifest/hash | Not persisted; must be generated before canonical reporting |
| Test manifest/hash | Not persisted; must be generated before canonical reporting |
| Split seed | 42; `os.listdir` is not sorted, so the seed alone is insufficient to reproduce sample membership across filesystems |

The split policy must be fixed before calibration. If multiple images of the same individual exist, repeated-identity dependence must be documented and preferably controlled through identity-level splitting.

## 9. Automated validation requirements

Tests must verify:

1. finite valid age for every retained sample;
2. deterministic parsing of the numeric target;
3. exact bin assignment at every internal threshold;
4. right-bin inclusion for threshold equality;
5. one and only one canonical class per retained target;
6. equality of supplied and derived labels when a derived-label interface is declared;
7. separation of \(Z\) and \(Y_{\mathrm{ord}}\) in every batch;
8. no overlap of split manifests under the selected split unit;
9. per-class counts for train, validation, calibration, and test;
10. stable manifest, configuration, and dataset-contract hashes.

## 10. Known limitations

- Exchangeability can be affected by repeated identities, collection bias, and demographic imbalance.
- A random image-level split may overstate generalization if identities recur across partitions.
- The current unsorted directory listing prevents a seed-only reproducibility claim until sorted or frozen manifests are used.

## 11. Completion checklist

- [x] Freeze age source and parsing rule.
- [x] Define valid age range and exclusions.
- [x] Freeze exact bin thresholds and class count.
- [ ] Decide whether an identity-level split is feasible.
- [x] Document preprocessing.
- [x] Record license and source URL.
- [ ] Generate stable manifests and hashes.
- [ ] Add target-label-bin consistency tests.
