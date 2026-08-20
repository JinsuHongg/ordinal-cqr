# IEEE BigData 2026 — Conference Results Plan

## Purpose

This document defines how the current OCQR experimental results should be used in the IEEE BigData 2026 conference paper.

Because the paper has an 11-page limit, the main paper should not include every available full result table. The preferred conference narrative is:

1. Main benchmark comparison across all datasets and baselines.
2. Compact rare/extreme-class coverage analysis.
3. One representative OCQR ablation table.

Detailed per-class and diagnostic results should remain in the repository for reproducibility, rebuttal, supplementary material if available, and the later journal version.

---

## Experimental Scope

### Datasets
- RetinaMNIST: ordinal diabetic-retinopathy severity grading.
- UTKFace: ordinal facial age estimation.
- Solar flare forecasting: highly imbalanced ordinal forecasting with a chronological future-test split.

### Main methods
- LAC
- APS
- OAPS
- COPOC
- OCQR

### OCQR ablations
- OCQR-Pooled
- OCQR-NoHull
- OCQR-NoFallback
- OCQR-Raw
- OCQR-Nonnegative
- OCQR

### Evaluation protocol
- Target miscoverage: alpha = 0.10
- Seeds: 0, 1, 2, 3, 4
- Report mean ± standard deviation across seeds.
- Keep train, validation, calibration, and test data separate.
- Select the fitted model/checkpoint before conformal calibration.
- Use frozen split manifests and the common evaluator for final results.

---

# Recommended Main-Paper Tables

## Table 1 — Main Benchmark Comparison

**Keep this table in the main paper.**

Recommended columns:

| Dataset | Method | Marginal Coverage | Macro Class-Conditional Coverage | Worst-Class Coverage | Mean Set Size | Full-Set Rate |
|---|---|---:|---:|---:|---:|---:|

### Current results

| Dataset | Method | Marginal (%) | Macro (%) | Worst (%) | Size | Full (%) |
|---|---|---:|---:|---:|---:|---:|
| RetinaMNIST | LAC | 89.7 ± 1.0 | 83.2 ± 1.8 | 63.0 ± 6.2 | 3.36 ± 0.10 | 29.5 ± 4.9 |
| RetinaMNIST | APS | 99.8 ± 0.4 | 99.6 ± 0.5 | 98.7 ± 1.9 | 4.91 ± 0.15 | 93.2 ± 10.4 |
| RetinaMNIST | OAPS | 90.0 ± 1.9 | 80.3 ± 2.7 | 37.0 ± 17.2 | 3.19 ± 0.10 | 15.8 ± 5.7 |
| RetinaMNIST | COPOC | 99.2 ± 0.5 | 98.5 ± 1.0 | 95.0 ± 3.5 | 4.74 ± 0.16 | 84.9 ± 9.2 |
| RetinaMNIST | OCQR | 95.8 ± 1.2 | 93.3 ± 1.3 | 81.4 ± 4.7 | 4.04 ± 0.18 | 62.3 ± 9.5 |
| UTKFace | LAC | 90.9 ± 0.4 | 83.9 ± 1.6 | 66.4 ± 9.0 | 1.60 ± 0.04 | 0.0 ± 0.0 |
| UTKFace | APS | 99.8 ± 0.2 | 99.6 ± 0.4 | 98.8 ± 1.0 | 3.41 ± 0.18 | 13.6 ± 7.2 |
| UTKFace | OAPS | 91.1 ± 0.4 | 84.3 ± 2.2 | 68.7 ± 11.6 | 1.62 ± 0.05 | 0.1 ± 0.1 |
| UTKFace | COPOC | 99.8 ± 0.1 | 99.5 ± 0.4 | 98.2 ± 1.6 | 3.38 ± 0.07 | 18.4 ± 3.6 |
| UTKFace | OCQR | 98.7 ± 0.2 | 98.1 ± 0.4 | 96.7 ± 0.8 | 2.35 ± 0.06 | 0.0 ± 0.1 |
| Solar flare | LAC | 87.1 ± 0.7 | 73.9 ± 2.7 | 10.0 ± 15.1 | 1.83 ± 0.05 | 0.0 ± 0.0 |
| Solar flare | APS | 98.5 ± 0.1 | 95.7 ± 1.5 | 81.9 ± 9.0 | 2.96 ± 0.04 | 0.6 ± 0.5 |
| Solar flare | OAPS | 86.6 ± 0.5 | 72.7 ± 1.7 | 5.8 ± 8.8 | 1.82 ± 0.04 | 0.0 ± 0.0 |
| Solar flare | COPOC | 97.0 ± 0.9 | 90.1 ± 4.8 | 55.6 ± 24.4 | 3.07 ± 0.18 | 0.3 ± 0.3 |
| Solar flare | OCQR | 89.6 ± 5.4 | 88.4 ± 6.2 | 73.0 ± 17.8 | 2.60 ± 0.11 | 0.0 ± 0.0 |

### Main interpretation

Interpret coverage jointly with prediction-set size and full-set rate.

- **UTKFace:** OCQR achieves `96.7 ± 0.8%` worst-class coverage with mean set size `2.35 ± 0.06`. APS and COPOC achieve slightly higher worst-class coverage, but with substantially larger mean sets (`3.41` and `3.38`).
- **RetinaMNIST:** APS and COPOC are highly conservative. APS has mean set size `4.91/5` and full-set rate `93.2%`; COPOC has mean set size `4.74/5` and full-set rate `84.9%`. OCQR reduces prediction-set size and full-set frequency, although its worst-class coverage is lower than APS and COPOC.
- LAC and OAPS can satisfy or approach the marginal target while providing much lower coverage for individual classes.
- Solar-flare results require separate interpretation because the chronological split does not establish exchangeability.

Do not claim that OCQR has the highest coverage on all datasets.

---

## Table 2 — Rare / Extreme-Class Coverage

Do **not** use three separate full class-wise tables in the main conference paper.

Instead, create one compact table that highlights the classes responsible for poor worst-class coverage.

Recommended format:

| Dataset | Class | Test Count | LAC | APS | OAPS | COPOC | OCQR |
|---|---|---:|---:|---:|---:|---:|---:|
| RetinaMNIST | 4 | 20 | 68.0 ± 10.4 | 100.0 ± 0.0 | 37.0 ± 17.2 | 95.0 ± 3.5 | 82.0 ± 5.7 |
| UTKFace | 4 | 67 | 67.2 ± 10.2 | 99.4 ± 0.8 | 69.3 ± 12.3 | 98.8 ± 1.9 | 97.0 ± 1.1 |
| Solar flare | X | 948 | 10.0 ± 15.1 | 81.9 ± 9.0 | 5.8 ± 8.8 | 55.6 ± 24.4 | 86.1 ± 6.6 |

Purpose of this table:

- Show which classes drive the worst-class metric.
- Demonstrate behavior that marginal coverage can hide.
- Directly support the motivation for class-conditional calibration.

Keep the complete per-class tables in experiment artifacts and the future journal version.

Do not claim that every rare class in every ordinal problem is necessarily the most severe or consequential class.

---

## Table 3 — OCQR Ablation

Use **RetinaMNIST as the representative ablation dataset** in the main conference paper.

A second full UTKFace ablation table is not necessary unless substantial page space remains.

### Current RetinaMNIST ablation

| Method | Marginal (%) | Macro (%) | Worst (%) | Size | CCR (%) | SFS | MDJ | Empty (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OCQR-Pooled | 96.0 ± 1.2 | 89.2 ± 2.3 | 53.0 ± 11.0 | 3.90 ± 0.11 | 96.0 ± 1.2 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 |
| OCQR-NoHull | 95.8 ± 1.2 | 93.3 ± 1.3 | 81.4 ± 4.7 | 4.04 ± 0.18 | 95.0 ± 1.7 | 1.01 ± 0.01 | 0.01 ± 0.01 | 0.0 ± 0.0 |
| OCQR-NoFallback | 95.8 ± 1.2 | 93.3 ± 1.3 | 81.4 ± 4.7 | 4.04 ± 0.18 | 95.8 ± 1.2 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 |
| OCQR-Raw | 95.8 ± 1.2 | 93.3 ± 1.3 | 81.4 ± 4.7 | 4.04 ± 0.18 | 95.0 ± 1.7 | 1.01 ± 0.01 | 0.01 ± 0.01 | 0.0 ± 0.0 |
| OCQR-Nonnegative | 96.2 ± 0.9 | 94.0 ± 1.1 | 82.0 ± 5.7 | 4.11 ± 0.17 | 96.2 ± 0.9 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 |
| OCQR | 95.8 ± 1.2 | 93.3 ± 1.3 | 81.4 ± 4.7 | 4.04 ± 0.18 | 95.8 ± 1.2 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 |

### Recommended conference columns

Prefer:

| Method | Marginal | Macro | Worst | Mean Set Size |
|---|---:|---:|---:|---:|

Only retain structural diagnostic columns when they contain informative nonzero behavior.

### Main ablation finding

The main empirical effect comes from **class-specific Mondrian calibration**.

The non-negative-correction variant is active on RetinaMNIST in every seed:
it clips one negative class correction, producing a small coverage increase
with a mean-set-size increase from `4.04 ± 0.18` to `4.11 ± 0.17`. It is
inactive on UTKFace and is therefore reported only in the RetinaMNIST table.

OCQR-Raw matches OCQR-NoHull on RetinaMNIST because no raw set is empty. It
leaves `0.7 ± 1.0%` of raw sets fragmented, confirming that hull closure is a
structural safeguard even when its aggregate coverage and size effects are
small.

- RetinaMNIST:
  - OCQR-Pooled worst-class coverage: `53.0 ± 11.0%`
  - OCQR worst-class coverage: `81.4 ± 4.7%`
- UTKFace:
  - OCQR-Pooled worst-class coverage: `87.2 ± 2.3%`
  - OCQR worst-class coverage: `96.7 ± 0.8%`

This supports the contribution that class-specific calibration improves class-level reliability relative to pooled calibration.

### Hull and fallback interpretation

Current results show:

- Empty-set fallback was not activated on RetinaMNIST or UTKFace.
- Removing the hull produces almost no change in aggregate metrics; RetinaMNIST shows only a very small structural effect.
- NoHull and NoFallback should therefore be described as **diagnostic variants**.
- The hull and fallback are **structural safeguards**, not components that must improve coverage or efficiency on every dataset.

Suitable wording:

> Class-specific calibration produces the main improvement in worst-class coverage, while the hull and fallback act as conservative structural safeguards and are rarely activated on the evaluated image benchmarks.

---

# Tables to Remove or Merge from the Main Paper

Avoid separate full-width main-paper tables for:

1. RetinaMNIST full class-wise coverage.
2. UTKFace full class-wise coverage.
3. Solar-flare full class-wise coverage.
4. UTKFace full OCQR ablation.

Do not delete these results from the repository.

Retain them for:

- reproducibility,
- internal validation,
- reviewer/rebuttal analysis,
- supplementary material if allowed,
- later journal submission.

---

# Solar-Flare Reporting Rule

The solar-flare dataset uses a **frozen chronological future-test split** with `n = 27,620`.

The X class has `n = 948`.

The chronological split evaluates future extrapolation. It does **not** establish the calibration--test exchangeability assumption used by the finite-sample OCQR coverage theorem.

Therefore:

- Report solar results as empirical temporal-extrapolation results.
- Do not claim that the solar experiment validates the theorem.
- Do not describe empirical solar coverage as a guaranteed conformal coverage result under distribution shift.

Current X-class coverage:

- LAC: `10.0 ± 15.1%`
- APS: `81.9 ± 9.0%`
- OAPS: `5.8 ± 8.8%`
- COPOC: `55.6 ± 24.4%`
- OCQR: `86.1 ± 6.6%`

This is an important empirical result, but it must remain separate from the formal exchangeability-based guarantee.

---

# Recommended Results Narrative

## 1. Coverage--efficiency tradeoff

Do not interpret high coverage independently of prediction-set size.

RetinaMNIST example:

- APS: marginal `99.8%`, size `4.91`, full-set `93.2%`
- COPOC: marginal `99.2%`, size `4.74`, full-set `84.9%`
- OCQR: marginal `95.8%`, size `4.04`, full-set `62.3%`

APS and COPOC achieve very high coverage but often produce sets containing most or all five classes.

UTKFace gives the clearest OCQR coverage--efficiency result:

- APS: worst `98.8%`, size `3.41`
- COPOC: worst `98.2%`, size `3.38`
- OCQR: worst `96.7%`, size `2.35`

## 2. Class-specific calibration is the main ablation result

The pooled ablation substantially lowers worst-class coverage while keeping marginal coverage relatively high.

Use this result to support the role of true-label Mondrian calibration.

## 3. Class-wise analysis explains marginal coverage

Use the compact rare/extreme-class table to show that aggregate marginal coverage can hide very low coverage for individual classes.

---

# Claim Discipline for Agent-Generated Manuscript Text

## Supported directions

The agent may write that:

- OCQR targets class-conditional coverage while preserving ordinal contiguity.
- Class-specific calibration improves worst-class coverage relative to the OCQR-Pooled ablation.
- OCQR produces smaller prediction sets than APS and COPOC on RetinaMNIST and UTKFace in the current results.
- APS and COPOC are substantially conservative on some benchmarks in the current experiments.
- Solar-flare results characterize empirical behavior under chronological future extrapolation.

## Avoid unsupported claims

Do not write:

- "OCQR achieves the best coverage on all datasets."
- "OCQR uniformly outperforms all baselines."
- "The solar experiment validates the finite-sample coverage guarantee."
- "The hull or fallback improves coverage on every dataset."
- "OCQR guarantees smaller prediction sets than APS or COPOC."
- "Rare ordinal classes are always the most consequential classes."

Prefer dataset-specific, empirical wording.

---

# Preferred Conference Results Structure

The main Results section should target the following structure:

1. **Main Benchmark Results**
   - one full-width table
   - discuss marginal, macro, worst-class coverage jointly with set size

2. **Rare / Extreme-Class Coverage**
   - one compact table
   - explain what drives poor worst-class coverage

3. **OCQR Ablation**
   - one representative RetinaMNIST table
   - emphasize pooled vs class-specific calibration
   - summarize hull/fallback behavior in text

This three-table structure is preferred over the current six full-width tables.

---

# Journal-Version Retention

Keep all detailed experiment outputs in the repository, including:

- full class-wise coverage for all datasets,
- RetinaMNIST and UTKFace ablations,
- structural diagnostics,
- raw-set diagnostics,
- calibration counts,
- class-specific conformal corrections,
- seed-level results.

These results may be useful for the full journal version even if they are omitted from the IEEE BigData conference paper.
