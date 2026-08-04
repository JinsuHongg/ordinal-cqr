---
dataset_id: eyepacs
name: EyePACS Diabetic Retinopathy Detection
card_version: "0.1.0"
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
  source: trainLabels.csv level field
class_count: 5
bins:
  convention: "B_k = [b_k, b_{k+1})"
  threshold_equality: right_bin
  thresholds: [0.5, 1.5, 2.5, 3.5]
split_policy: "stratified image-level 60/10/20/10 split; seed 42"
license: "Kaggle competition data; verify terms before redistribution"
source_url: "https://www.kaggle.com/c/diabetic-retinopathy-detection"
last_updated: "2026-08-03"
---

# EyePACS Dataset Card and OCQR Metadata

## 1. Purpose

EyePACS is a five-level diabetic-retinopathy benchmark. The reviewed adapter has no separate continuous clinical measurement, so the canonical OCQR representation is the class-index embedding \(Z=Y_{\mathrm{ord}}\). This is a surrogate ordinal coordinate, not a measured disease-severity scale.

## 2. Canonical interface and labels

Every sample exposes \((X,Z,Y_{\mathrm{ord}})\), where `image` from `trainLabels.csv` identifies the fundus image and `level` supplies the ordinal label. The current mapping is identity:

| Index | `level` | Meaning | Retained |
|---:|---:|---|---|
| 0 | 0 | No diabetic retinopathy | Yes |
| 1 | 1 | Mild diabetic retinopathy | Yes |
| 2 | 2 | Moderate diabetic retinopathy | Yes |
| 3 | 3 | Severe diabetic retinopathy | Yes |
| 4 | 4 | Proliferative diabetic retinopathy | Yes |

The bins are \([ -\infty,0.5)\), \([0.5,1.5)\), \([1.5,2.5)\), \([2.5,3.5)\), and \([3.5,+\infty)\). Thus \(Z=Y_{\mathrm{ord}}\) satisfies the target-label-bin consistency relation for all five labels.

## 3. Reviewed local source artifact

The reviewed `trainLabels.csv` has 35,126 rows with columns `image` and `level`, SHA-256 `1dd6600d01dd0ff1e42ce840ffa501ae32b346745ff4c4f9296e62a2e9632f88`. The corresponding reviewed local image directory contains all listed `.jpeg` files.

| Class | Count |
|---:|---:|
| 0 | 25,810 |
| 1 | 2,443 |
| 2 | 5,292 |
| 3 | 873 |
| 4 | 708 |

These counts describe the reviewed source CSV before splitting, not a frozen experiment manifest.

## 4. Preprocessing and sampling

| Component | Current adapter behavior |
|---|---|
| Image path | Uses `image`; appends `.jpeg` when no recognized image suffix is present |
| Decoding | Pillow `Image.open(...).convert("RGB")` |
| Resize | 128 × 128 |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Training augmentation | Random horizontal flip |
| Training sampler | Inverse-frequency `WeightedRandomSampler` with replacement |
| Invalid image handling | No preflight validation in the adapter; decode errors raise |
| Patient/eye identity handling | Not implemented; filename laterality is not used to group splits |

The adapter adds a singleton time dimension after image preprocessing to match the model interface.

## 5. Split and provenance

The current data module performs stratified image-level splitting with seed 42: 60% train, 10% validation, 20% calibration, and 10% test. It does not persist the resulting split manifests. Patient-level and left/right-eye leakage controls are therefore not established.

Canonical reporting requires saved manifests and hashes, an overlap audit at the declared patient/eye split unit, and target-label-bin validation for every retained row.

## 6. Known limitations

- The numeric coordinate is a class-index embedding.
- The source CSV is extremely imbalanced, especially in classes 3 and 4.
- The present image-level split can place correlated left/right eyes or patients across partitions.
- Source license and access terms must be checked against the current competition-data agreement before redistribution.

## 7. Completion checklist

- [x] Define the class-index representation and midpoint thresholds.
- [x] Record reviewed source schema, class counts, and source-file hash.
- [x] Document preprocessing and weighted sampling.
- [ ] Persist train/validation/calibration/test manifests and hashes.
- [ ] Add patient/eye-level overlap and leakage audits.
- [ ] Verify redistribution terms for the exact source release.
