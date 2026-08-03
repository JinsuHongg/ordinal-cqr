# Ordinal Conformalized Quantile Regression (OCQR): Project Summary

**Project:** Ordinal Conformal Prediction
**Method:** Ordinal Conformalized Quantile Regression (OCQR)
**Canonical version:** 0.3.0
**Status:** Research and implementation project
**Primary documents:** `ocqr_theory.md`, `ocqr_contract.md`

---

## 1. Project objective

This project develops **Ordinal Conformalized Quantile Regression (OCQR)**, a conformal prediction framework for ordinal classification. The method is designed to produce prediction sets that:

- satisfy a finite-sample **class-conditional ordinal coverage** guarantee under the stated assumptions;
- respect the natural order of the labels;
- are nonempty and ordinally contiguous after post-processing;
- support both class-only ordinal datasets and datasets with an observed continuous target;
- remain applicable under severe class imbalance through true-label Mondrian calibration.

The central idea is to connect continuous-target conformalized quantile regression with discrete ordinal prediction. A lower and upper quantile model predicts a numeric interval for a target \(Z\). Class-specific conformal corrections are then calibrated using the supplied true ordinal labels. Each ordinal candidate is included when its class-specific numeric acceptance interval intersects the numeric bin assigned to that class.

---

## 2. Problem setting

Each retained observation must expose

\[
(X,Z,Y_{\mathrm{ord}}),
\]

where:

- \(X\in\mathcal X\) is the model input;
- \(Z\in\mathbb R\) is the numeric target used for quantile regression and the conformal score;
- \(Y_{\mathrm{ord}}\in\{0,\ldots,K-1\}\) is the supplied ordinal label used for Mondrian grouping;
- \(B_k=[b_k,b_{k+1})\) is the fixed numeric bin associated with class \(k\), with

\[
-\infty=b_0<b_1<\cdots<b_K=+\infty.
\]

The canonical target-label consistency requirement is

\[
Y_{\mathrm{ord}}=k \Longrightarrow Z\in B_k.
\]

For canonical disjoint and exhaustive bins, retained samples should satisfy the equivalent relationship

\[
Y_{\mathrm{ord}}=k \Longleftrightarrow Z\in B_k.
\]

The implementation must accept \(Z\) and \(Y_{\mathrm{ord}}\) as logically distinct values, even when \(Z\) is deterministically embedded from the ordinal label.

---

## 3. Canonical OCQR pipeline

### 3.1 Quantile prediction

A common model predicts lower and upper conditional quantiles of \(Z\):

\[
\widehat q_L(x),\qquad \widehat q_U(x).
\]

Canonical experiments use quantile levels

\[
\tau_L=\alpha/2,
\qquad
\tau_U=1-\alpha/2,
\]

although these levels are modeling choices rather than requirements for conformal validity.

### 3.2 Crossing correction

Quantile crossing is corrected deterministically:

\[
\widehat\ell(x)=\min\{\widehat q_L(x),\widehat q_U(x)\},
\qquad
\widehat u(x)=\max\{\widehat q_L(x),\widehat q_U(x)\}.
\]

All downstream calibration and prediction operations use these ordered endpoints.

### 3.3 CQR nonconformity score

The canonical score is

\[
s(x,z)=\max\{\widehat\ell(x)-z,\;z-\widehat u(x)\}.
\]

Negative scores are valid and must not be clipped. For a correction \(q\), the numeric acceptance set is

\[
A_q(x)=\{z:s(x,z)\le q\}.
\]

For finite \(q\), exact inversion gives

\[
A_q(x)=
\begin{cases}
[\widehat\ell(x)-q,\widehat u(x)+q],
&\widehat\ell(x)-q\le\widehat u(x)+q,\\
\varnothing,&\text{otherwise.}
\end{cases}
\]

For \(q=+\infty\), \(A_q(x)=\mathbb R\).

### 3.4 True-label Mondrian calibration

For class \(k\), calibration scores are grouped using the supplied true ordinal label:

\[
\mathcal I_k=\{i:Y_i=k\},
\qquad
N_k=|\mathcal I_k|.
\]

The requested conformal rank is

\[
r_k=\left\lceil(N_k+1)(1-\alpha)\right\rceil.
\]

The class scores are augmented with one \(+\infty\) value, and the correction is the exact augmented order statistic

\[
q_k=\widetilde S_{k,(r_k)}.
\]

This definition handles empty or insufficient class-specific calibration groups conservatively. When no finite order statistic supports the requested rank, \(q_k=+\infty\).

### 3.5 Candidate-specific ordinal membership

Each candidate class is evaluated using its own correction:

\[
I_k(x)=A_{q_k}(x).
\]

The raw ordinal set is

\[
C_{\mathrm{raw}}(x)
=
\{k:I_k(x)\cap B_k\neq\varnothing\}.
\]

A correction associated with a point-predicted class must not be reused for all candidates.

### 3.6 Empty-set fallback and ordinal hull

If the raw set is empty, it is replaced by the full label space:

\[
\widetilde C_{\mathrm{raw}}(x)=
\begin{cases}
C_{\mathrm{raw}}(x),&C_{\mathrm{raw}}(x)\neq\varnothing,\\
\mathcal Y,&C_{\mathrm{raw}}(x)=\varnothing.
\end{cases}
\]

The final prediction set is the ordinal hull

\[
C(x)=\operatorname{Hull}(\widetilde C_{\mathrm{raw}}(x)).
\]

The final output is therefore nonempty and contiguous. Both fallback and hull are add-only operations and preserve coverage, although they can reduce efficiency.

---

## 4. Theoretical claim

Under the following conditions:

1. the fitted model and all method choices are frozen using training and validation data only;
2. calibration and test data are independent of model and method selection;
3. calibration scores and the test score are exchangeable within each true-label class under the stated conditioning;
4. target-label-bin consistency holds;
5. Mondrian groups use the supplied true ordinal label;
6. the exact augmented order statistic is used;
7. every candidate class is evaluated with its own \(q_k\);
8. candidate membership follows numeric interval-bin intersection;
9. fallback and hull only add labels;

OCQR satisfies, for each ordinal class \(k\),

\[
\Pr\{Y_{n+1}\in C(X_{n+1})\mid \mathcal F,Y_{n+1}=k\}
\ge 1-\alpha
\]

for almost every frozen fitted procedure \(\mathcal F\). Averaging over \(\mathcal F\) gives marginal label-conditional coverage:

\[
\Pr\{Y_{n+1}\in C(X_{n+1})\mid Y_{n+1}=k\}
\ge 1-\alpha.
\]

This guarantee is conditional on the method assumptions. In particular, a chronological future holdout does not itself establish exchangeability.

---

## 5. Dataset scope

The project uses four ordinal prediction datasets:

1. MedMNIST RetinaMNIST;
2. UTKFace;
3. Adience;
4. solar flare prediction data.

Each dataset requires a separate versioned dataset contract specifying the source fields, target representation, label definition, thresholds, boundary rules, invalid-sample handling, split provenance, and automated consistency tests.

### 5.1 Dataset representation summary

| Dataset | Input \(X\) | Numeric target \(Z\) | Ordinal label \(Y_{\mathrm{ord}}\) | Representation type | Required dataset-contract decision |
|---|---|---|---|---|---|
| RetinaMNIST | Retinal fundus image | Canonical class-index embedding \(Z=Y_{\mathrm{ord}}\) | Ordered retinal severity label | Class-only surrogate coordinate | Confirm official label ordering, class count, midpoint thresholds, retained-sample rules, and split hashes |
| UTKFace | Face image | Chronological age | Fixed age-bin index | Observed numeric target | Define exact age-bin thresholds, boundary convention, valid age range, filtering rules, and split construction |
| Adience | Face image | Dataset-contract-dependent age coordinate | Ordered age-group label | Ordinal age-group dataset; canonical numeric interface must be declared | Decide whether \(Z\) is a fixed class embedding or another documented numeric representation; define all bins and consistency rules |
| Solar flare prediction | Solar active-region or multimodal observational input | Peak X-ray flux or a fixed deterministic transformation | Ordered flare class | Observed numeric target | Define flux transformation, A/B/C/M/X thresholds, supplied-label mapping, no-flare handling, temporal split, active-region dependence, and validation artifact |

### 5.2 RetinaMNIST

RetinaMNIST is treated as a class-only ordinal dataset. The canonical OCQR representation is

\[
Z=Y_{\mathrm{ord}},
\]

with fixed midpoint bins around the embedded class indices. This coordinate is a surrogate ordinal representation rather than a measured continuous disease-severity value.

The dataset contract must document:

- the exact number and order of classes;
- the mapping from source labels to \(0,\ldots,K-1\);
- midpoint thresholds used to define \(B_k\);
- integer-safe and floating-point-safe boundary behavior;
- any sample exclusion or preprocessing;
- training, validation, calibration, and test manifests or hashes.

Alternative monotone embeddings must be treated as separately named ablations because candidate membership and efficiency are not invariant to recoding.

### 5.3 UTKFace

UTKFace uses observed chronological age as \(Z\). The ordinal label is the index of a fixed age interval:

\[
Y_{\mathrm{ord}}=k
\Longleftrightarrow
Z\in B_k.
\]

The dataset contract must define:

- the exact age-bin thresholds;
- whether age is represented as integer years or another fixed numeric convention;
- right-bin inclusion at every internal threshold;
- handling of invalid, missing, or implausible ages;
- image filtering and identity or duplication considerations;
- split generation and provenance.

The continuous age target provides a direct continuous-to-discrete bridge: conformal coverage of age implies inclusion of the true age bin, while bin coverage can be more conservative.

### 5.4 Adience

Adience provides ordered age-group labels. The two normative documents do not specify a canonical Adience target representation, so the project must define it in a dedicated dataset contract before canonical experiments.

A defensible default is a fixed class-only embedding

\[
Z=e(Y_{\mathrm{ord}}),
\]

with the canonical choice \(e(k)=k\) and midpoint bins. This option avoids pretending that an age-group label is an exact chronological age. Any alternative representation, such as interval midpoints or another age coordinate, must be explicitly justified, versioned, and reported as a separate method variant.

The Adience contract must specify:

- the ordered source age groups and their mapping to contiguous indices;
- the chosen numeric representation \(Z\);
- exact thresholds and boundary convention;
- treatment of overlapping, ambiguous, or open-ended age ranges;
- subject-level splitting to control identity leakage where applicable;
- invalid and missing metadata handling;
- representation limitations.

### 5.5 Solar flare prediction dataset

The solar flare dataset uses peak X-ray flux, or a fixed deterministic transformation of it, as \(Z\). The supplied flare class is used as \(Y_{\mathrm{ord}}\), with the ordinal sequence defined by the dataset contract, such as A, B, C, M, and X when all five classes are retained.

The calibration grouping must use the supplied flare class, while the numeric flux is used in pinball loss and the conformal score. A reproducible validation artifact must verify target-label-bin consistency for every retained observation.

The dataset contract must specify:

- the exact flux field and units;
- any logarithmic or other deterministic transformation;
- all flare-class thresholds including \(\pm\infty\) after transformation;
- treatment of no-flare observations and whether they form a separate ordinal class;
- event association and forecast-window construction;
- active-region grouping and repeated-observation dependence;
- chronological development and future-holdout windows;
- retained-sample manifests, split hashes, or explicit documentation when stable hashes are unavailable.

A temporal split can prevent direct leakage and evaluate future extrapolation, but it does not prove within-class exchangeability. The solar study must therefore report both theorem-based claims under the exchangeability assumption and empirical performance under temporal distribution shift.

---

## 6. Experimental design

### 6.1 Data partitions

Every experiment must separate:

- **training data:** model fitting;
- **validation data:** checkpoint, hyperparameter, and variant selection;
- **calibration data:** prespecified computation of \(q_k\) only;
- **test data:** final evaluation only.

Before calibration labels or targets are accessed, the following must be frozen:

- model parameters and checkpoint;
- target transformation and embedding;
- bin thresholds and boundary convention;
- quantile levels;
- nonconformity score;
- hyperparameters;
- Mondrian taxonomy;
- fallback and hull policies;
- metric definitions;
- canonical method version.

Calibration results must not be used to choose among models, embeddings, bins, scores, or post-processing policies.

### 6.2 Core baselines and ablations

The final experiment plan should distinguish canonical OCQR from explicitly named variants. Relevant comparisons include:

- uncalibrated quantile-bin prediction;
- global CQR without Mondrian calibration;
- true-label Mondrian OCQR;
- ordinal methods that construct sets directly from class probabilities;
- nominal conformal prediction baselines;
- OCQR before and after hull closure;
- alternative class embeddings for class-only datasets;
- alternative fixed quantile levels;
- alternative model backbones, selected using training and validation data only.

No ablation may be presented as canonical OCQR unless it satisfies version 0.3.0 of the theory and method contract.

---

## 7. Evaluation metrics

### 7.1 Coverage

Required coverage metrics include:

- overall marginal coverage;
- per-class label-conditional coverage;
- worst-class coverage;
- raw-set coverage;
- final-set coverage;
- continuous-target interval coverage for datasets with observed numeric \(Z\), when applicable.

### 7.2 Efficiency and ordinal structure

Required efficiency and structural metrics include:

- average final set size;
- per-class average set size;
- singleton rate;
- full-set rate;
- raw empty-set rate;
- fragmented raw-set rate;
- hull inflation;
- fallback inflation;
- total inflation from raw to final set;
- interval width for observed numeric-target datasets.

Coverage and efficiency must be reported together. A class with \(q_k=+\infty\) has conservative coverage but no informative finite correction.

### 7.3 Dataset-specific analysis

Additional analyses should include:

- class-frequency and calibration-count distributions;
- correction values \(q_k\) by class;
- sensitivity to rare-class calibration size;
- age-bin boundary behavior for UTKFace and Adience;
- class-embedding sensitivity for RetinaMNIST and possibly Adience;
- temporal degradation, active-region dependence, and rare-event performance for solar flares.

---

## 8. Required calibration metadata and provenance

For every class and run, store:

- method name and version;
- dataset-contract identifier and version;
- class identifier;
- \(N_k\);
- requested rank \(r_k\);
- whether \(q_k\) is finite;
- \(q_k\), preserving `+inf` explicitly;
- number of ties at a finite \(q_k\);
- minimum and maximum finite calibration score;
- random seed;
- checkpoint identifier;
- configuration hash;
- code commit;
- split hash or retained-sample manifest hash;
- target transformation and bin specification.

Legacy outputs with incompatible method versions or metric schemas must not be mixed into canonical result tables.

---

## 9. Required implementation tests

The implementation test suite must cover:

1. exact score inversion;
2. pinball-loss formula and configured quantile levels;
3. raw quantile crossing and deterministic endpoint sorting;
4. negative nonconformity scores;
5. negative finite corrections;
6. NaN and infinite endpoint rejection;
7. finite target validation;
8. exact augmented order statistics;
9. tied calibration scores;
10. \(N_k=0\);
11. unattainable finite ranks producing \(+\infty\);
12. true-label grouping distinct from target-derived grouping;
13. candidate-specific corrections;
14. empty numeric acceptance intervals;
15. numeric interval-bin intersection;
16. integer targets with floating-point thresholds;
17. equality behavior at every internal bin threshold;
18. fragmented raw sets;
19. ordinal hull inclusion and contiguity;
20. empty raw sets producing the full label space;
21. raw-empty, hull-inflation, fallback-inflation, fragmentation, and full-set metrics;
22. random-\(N_k\) synthetic per-class coverage;
23. target-label-bin consistency for every dataset contract.

---

## 10. Implementation alignment

The current project specification identifies the following implementation areas:

- class-index and bin membership logic;
- full-label empty-set fallback;
- separate \((X,Z,Y_{\mathrm{ord}})\) batch interface;
- true-label Mondrian calibration;
- pinball-loss training;
- crossing correction;
- explicit nonfinite endpoint and target rejection;
- strict calibration metadata persistence;
- raw, final, hull, fallback, fragmentation, and full-set metrics.

Canonical configuration must include:

```yaml
method:
  name: ocqr
  version: "0.3.0"

prediction:
  empty_candidate_set: full_label_space
```

Configurations using an earlier method version or a `preserve_empty` policy are not canonical OCQR v0.3.

---

## 11. Main research questions

The project is organized around the following questions:

1. Can continuous-target quantile regression be converted into valid and contiguous ordinal prediction sets?
2. Does true-label Mondrian calibration improve per-class reliability under ordinal class imbalance?
3. What efficiency cost is introduced by candidate-specific correction, hull closure, and full-set fallback?
4. How does OCQR behave differently on class-only surrogate embeddings and observed numeric targets?
5. How sensitive are RetinaMNIST and Adience results to the chosen ordinal embedding?
6. How do age-bin definitions and boundary conventions affect UTKFace and Adience prediction sets?
7. How robust is OCQR under the severe imbalance and temporal distribution shift of solar flare forecasting?
8. When rare classes produce \(+\infty\) corrections, how should coverage and efficiency be interpreted and reported?

---

## 12. Defensible manuscript claim

> Under conditional within-class exchangeability, a fixed target-to-bin representation, exact true-label Mondrian split-conformal calibration, candidate-specific score inversion, and add-only fallback and hull post-processing, OCQR provides finite-sample class-conditional ordinal coverage. The full-set empty-set fallback guarantees nonempty contiguous outputs, while potentially reducing efficiency.

For chronological solar flare experiments, this statement must be qualified: the theorem is evaluated under its exchangeability assumption, while temporal extrapolation is assessed empirically rather than guaranteed by the split design.

---

## 13. Immediate project deliverables

The next project artifacts should be:

1. a versioned dataset contract for RetinaMNIST;
2. a versioned dataset contract for UTKFace;
3. a versioned dataset contract for Adience;
4. a versioned dataset contract and consistency-validation artifact for solar flare data;
5. stable train/validation/calibration/test manifests or documented hashes;
6. a synchronized canonical configuration for OCQR v0.3.0;
7. deterministic contract tests and synthetic coverage tests;
8. a unified evaluation schema for coverage, efficiency, fallback, hull, and provenance metrics;
9. a manuscript method section that matches the theory, contract, code, and reported experiments.

---

## 14. Open decisions

The following decisions are not resolved by the two normative source documents and must be fixed before canonical experiments:

- the exact UTKFace age bins;
- the canonical numeric representation and bins for Adience;
- the exact RetinaMNIST source-label mapping and class count used by the implementation;
- the solar flare flux transformation and retained class taxonomy;
- treatment of no-flare samples in the solar dataset;
- dataset-specific split construction and leakage controls;
- stable dataset-contract identifiers and manifest hashes.

These choices must not be selected using calibration or test performance.
