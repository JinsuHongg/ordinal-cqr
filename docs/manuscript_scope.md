# Manuscript Scope and Content Allocation Guide

## Purpose

This document defines the scope, content, claims, and experimental responsibilities of two manuscripts developed from the same OCQR project.

The manuscript names are intentionally generic:

- **Conference-track manuscript**: a concise paper centered on the core method, the main theoretical result, and focused empirical validation.
- **Journal-track manuscript**: a substantially extended paper with broader theoretical development, deeper experiments, additional analyses, and complete implementation and reproducibility details.

This file is intended for authors and AI coding/writing agents working in the project repository. Before drafting, editing, or moving content, determine which manuscript track the content belongs to by following the rules below.

---

## 1. Shared Scientific Core

Both manuscripts study **Ordinal Conformalized Quantile Regression (OCQR)** for ordinal classification.

The shared core includes:

1. A numeric target representation \(Z\) associated with an ordinal label
   \[
   Y \in \mathcal{Y}=\{0,\ldots,K-1\}.
   \]

2. Ordered bins
   \[
   B_k=[b_k,b_{k+1}),
   \]
   with target--label consistency
   \[
   Y=k \Longrightarrow Z\in B_k.
   \]

3. Lower and upper quantile predictors trained using pinball loss.

4. The CQR nonconformity score
   \[
   s(x,z)=\max\{\widehat\ell(x)-z,\;z-\widehat u(x)\}.
   \]

5. True-label Mondrian calibration with a class-specific conformal correction \(q_k\).

6. Candidate-wise test-time evaluation:
   candidate class \(k\) is evaluated using its own \(q_k\).

7. Candidate inclusion through numeric interval--bin intersection:
   \[
   C_{\mathrm{raw}}(x)
   =
   \{k:A_{q_k}(x)\cap B_k\neq\varnothing\}.
   \]

8. Conservative add-only post-processing:
   - full-label fallback when the raw set is empty;
   - ordinal hull to obtain a contiguous final set.

9. Finite-sample class-conditional ordinal coverage under the stated assumptions.

The two manuscripts may share these scientific ideas, notation, and method identity, but they must differ substantially in depth, scope, experiments, and analysis.

---

## 2. High-Level Difference Between the Two Manuscripts

| Dimension | Conference-Track Manuscript | Journal-Track Manuscript |
|---|---|---|
| Main goal | Present the core OCQR idea clearly and validate it on focused benchmarks | Provide the complete theoretical, empirical, and methodological treatment |
| Length | Concise | Extended |
| Theory | One main theorem and a short proof or proof sketch | Full formal development with propositions, lemmas, theorem, corollary, assumptions, and edge cases |
| Experiments | Focused comparison on selected datasets and baselines | Broader benchmark suite, stronger ablations, sensitivity studies, robustness, and implementation validation |
| Related work | Limited to methods directly compared or needed to motivate OCQR | Broader and more systematic literature review |
| Method detail | Core algorithm and essential notation | Full method specification, variants, boundary cases, and implementation contract |
| Results | Main coverage and efficiency findings | Main results plus detailed diagnostic, subgroup, sensitivity, and failure analyses |
| Reproducibility | Essential setup details | Full data contracts, split provenance, calibration metadata, tests, and reproducibility artifacts |
| Positioning | A focused method paper | A complete reference paper for the method |

---

## 3. Conference-Track Manuscript

### 3.1 Primary Objective

The conference-track manuscript should answer the following question:

> Can a model-agnostic quantile-regression-based conformal method construct contiguous ordinal prediction sets while providing class-conditional coverage under class imbalance?

The paper should emphasize:

- the practical problem of fragmented ordinal prediction sets;
- the weakness of marginal coverage for rare classes;
- the candidate-specific class-conditional calibration mechanism;
- the coverage-preserving ordinal hull;
- focused empirical evidence.

It should not attempt to document every theoretical and implementation detail of the full project.

---

### 3.2 Recommended Title Style

Use a method-specific title that distinguishes this manuscript from the broader journal-track paper.

Recommended style:

> **OCQR: Class-Conditional Conformalized Quantile Regression for Ordinal Classification**

The title should emphasize the specific method rather than claim to cover the entire research area.

---

### 3.3 Recommended Section Structure

```latex
\section{Introduction}

\section{Related Work}
\subsection{Conformal Prediction for Classification}
\subsection{Class-Conditional Conformal Prediction}
\subsection{Conformal Prediction for Ordinal Classification}

\section{Problem Formulation}

\section{Ordinal Conformalized Quantile Regression}
\subsection{Quantile Regression for Ordinal Targets}
\subsection{Class-Conditional Conformal Calibration}
\subsection{Ordinal Prediction Set Construction}
\subsection{Coverage Guarantee}

\section{Experimental Setup}
\subsection{Datasets}
\subsection{Baselines}
\subsection{Implementation Details}
\subsection{Evaluation Metrics}

\section{Results and Discussion}
\subsection{Overall Performance}
\subsection{Class-Conditional Coverage}
\subsection{Prediction-Set Efficiency}
\subsection{Ablation Study}

\section{Conclusion}
```

---

### 3.4 Introduction Scope

The Introduction should include:

1. Why deterministic point predictions are insufficient in high-stakes ordinal tasks.
2. Why ordinal labels should not be treated as nominal categories.
3. Why standard classification CP can produce fragmented prediction sets.
4. Why marginal coverage can hide undercoverage of rare classes.
5. A concise description of OCQR.
6. A short contribution list.
7. A focused description of the selected benchmark domains.

The Introduction should not include:

- a long derivation of CQR;
- a full review of all ordinal CP variants;
- detailed solar physics background;
- implementation-level fallback logic;
- lengthy discussion of all theorem assumptions;
- a long paper-organization paragraph if page space is limited.

---

### 3.5 Related Work Scope

Include only methods that are:

- used as baselines;
- directly necessary to motivate OCQR;
- essential to explain the statistical guarantee.

Core coverage:

- LAC;
- APS;
- CQR;
- Mondrian or class-conditional CP;
- OAPS;
- COPOC.

Do not include unrelated methods solely for completeness.

Unless they are added to the experiments or become necessary for positioning, exclude:

- min-CPS;
- min-RCPS;
- conformal risk control;
- broad Bayesian uncertainty methods;
- ensemble uncertainty methods;
- unrelated regression uncertainty methods;
- extensive approximate conditional coverage literature.

The Related Work section should explain only:

- what each baseline does;
- what coverage objective it targets;
- whether it uses ordinal structure;
- whether it requires a specialized prediction model;
- how it differs from OCQR.

---

### 3.6 Theory Scope

The conference-track manuscript should include:

#### Required

- setup and notation;
- target--label consistency;
- quantile predictor definition;
- crossing correction;
- nonconformity score;
- class-specific calibration;
- exact augmented order statistic;
- candidate-specific interval;
- bin-intersection rule;
- ordinal hull;
- empty-set fallback;
- one class-conditional coverage theorem;
- a concise proof or proof sketch.

#### May be abbreviated

- conditioning on the fitted procedure \(\mathcal F\);
- random class calibration counts \(N_k\);
- tie handling;
- \(q_k=+\infty\);
- distinction between conditional-on-\(\mathcal F\) and marginal label-conditional coverage.

These points must remain mathematically correct, but they do not need separate propositions and lemmas in the main paper.

#### Exclude from the main text

- full score-inversion proposition;
- separate proofs for every lemma;
- long edge-case analysis;
- implementation error behavior;
- dtype and threshold-boundary rules;
- formal configuration and metadata requirements;
- exhaustive deterministic testing requirements.

These belong to the journal-track manuscript, appendix, supplementary material, or repository documentation.

---

### 3.7 Dataset Scope

The planned conference-track datasets are:

1. **Wine Quality**
   - tabular ordinal benchmark;
   - numeric quality score;
   - useful for evaluating OCQR on structured tabular data.

2. **UTKFace**
   - facial age estimation;
   - continuous age target mapped to ordinal age bins;
   - useful for evaluating a natural numeric target.

3. **RetinaMNIST**
   - medical image ordinal classification;
   - class-only ordinal representation;
   - useful for evaluating fixed ordinal embeddings.

4. **Solar Flare**
   - strongly imbalanced ordinal forecasting task;
   - numeric peak X-ray flux or a fixed transformation;
   - useful for evaluating rare extreme classes.

Do not describe these datasets as “big data” solely because they appear in a big-data-oriented venue.

Safer positioning:

- heterogeneous ordinal data;
- multiple modalities;
- severe class imbalance;
- rare-class uncertainty;
- model-agnostic post-hoc calibration;
- efficient candidate-wise conformal inference.

Do not use “large-scale” or “scalable” unless supported by sample-size, runtime, or complexity evidence.

---

### 3.8 Baseline Scope

#### Main baselines

- LAC;
- APS;
- OAPS;
- COPOC.

#### Calibration and method comparisons

- pooled or global CQR;
- globally calibrated OCQR, if it is technically distinct from pooled CQR.

#### OCQR ablations

- OCQR without class-specific calibration;
- OCQR without ordinal hull;
- optional OCQR without empty-set fallback, only if it is evaluated safely and clearly labeled as a noncanonical diagnostic variant.

Avoid including two baselines that are mathematically identical under different names.

Before retaining both “global OCQR” and “pooled CQR,” verify that they differ in at least one of:

- interval construction;
- calibration score;
- candidate inclusion rule;
- bin mapping;
- post-processing.

---

### 3.9 Evaluation Metrics

The main tables should prioritize:

- marginal coverage;
- per-class coverage;
- worst-class coverage;
- mean class-conditional coverage;
- average prediction-set size;
- full-set rate;
- fragmented raw-set rate or contiguity rate.

Ablation or supplementary tables may include:

- raw empty-set rate;
- hull inflation;
- fallback inflation;
- singleton rate;
- per-class set size;
- calibration support \(N_k\);
- finite versus infinite \(q_k\).

The central empirical question is not whether OCQR maximizes average coverage. It is whether OCQR improves coverage consistency across classes while maintaining reasonable prediction-set efficiency.

---

### 3.10 Claims Allowed in the Conference Track

Allowed when supported:

- OCQR is model-agnostic with respect to the backbone prediction architecture.
- OCQR uses class-specific true-label Mondrian calibration.
- Each candidate class is evaluated using its own class-specific correction.
- Add-only hull post-processing preserves coverage.
- OCQR provides finite-sample class-conditional coverage under the stated assumptions.
- OCQR produces nonempty contiguous prediction sets under the canonical fallback and hull policy.
- OCQR improves empirical class-conditional coverage consistency over evaluated baselines.
- OCQR maintains competitive prediction-set efficiency.

Avoid or qualify:

- “large-scale”;
- “scalable” without runtime or complexity experiments;
- “all existing ordinal CP methods provide only marginal coverage”;
- “distribution-free” without stating exchangeability;
- “conditional coverage” without specifying class-conditional or label-conditional coverage;
- causal claims about why performance improved;
- statements that empirical test coverage must always exceed the nominal level.

---

## 4. Journal-Track Manuscript

### 4.1 Primary Objective

The journal-track manuscript should serve as the complete methodological reference for OCQR.

It should answer:

> What is the full theoretical, algorithmic, empirical, and implementation framework required to establish and evaluate class-conditional conformal prediction for ordinal classification?

The journal-track manuscript should be broader than the conference-track manuscript in every major dimension:

- stronger theory;
- more complete literature positioning;
- more datasets or more extensive experiments;
- deeper ablation and sensitivity analysis;
- implementation validation;
- limitations and failure modes;
- reproducibility artifacts.

A title change alone is not sufficient to distinguish the manuscripts.

---

### 4.2 Recommended Title Style

Use a broader problem-level title rather than a narrow method acronym.

Recommended style:

> **Class-Conditional Conformal Prediction for Ordinal Classification**

This title positions the paper as a broader methodological treatment rather than only an introduction to one named algorithm.

---

### 4.3 Recommended Section Structure

```latex
\section{Introduction}

\section{Related Work}
\subsection{Conformal Prediction for Classification}
\subsection{Conditional and Group-Aware Conformal Prediction}
\subsection{Conformalized Quantile Regression}
\subsection{Ordinal Classification and Ordinal Uncertainty Quantification}
\subsection{Conformal Prediction for Ordinal Classification}

\section{Problem Setup}

\section{Proposed Framework}
\subsection{Numeric Representation of Ordinal Targets}
\subsection{Quantile Regression Model}
\subsection{Nonconformity Score and Exact Inversion}
\subsection{True-Label Mondrian Calibration}
\subsection{Candidate-Specific Ordinal Membership}
\subsection{Fallback and Ordinal Hull}

\section{Theoretical Analysis}
\subsection{Exchangeability Assumptions}
\subsection{Class-Specific Split-Conformal Validity}
\subsection{Numeric-to-Ordinal Inclusion}
\subsection{Coverage-Preserving Post-Processing}
\subsection{Main Class-Conditional Coverage Theorem}
\subsection{Discussion of Assumptions and Edge Cases}

\section{Experimental Design}
\subsection{Datasets and Dataset Contracts}
\subsection{Baselines}
\subsection{Implementation Details}
\subsection{Evaluation Metrics}
\subsection{Reproducibility Protocol}

\section{Main Results}

\section{Ablation and Sensitivity Analysis}

\section{Robustness, Limitations, and Failure Modes}

\section{Conclusion}
```

---

### 4.4 Theory Scope

The journal-track manuscript should contain the full formal theory.

Required elements:

1. Definition of the fitted-procedure sigma-field \(\mathcal F\).
2. Explicit separation of training, validation, calibration, and test data.
3. Formal score inversion.
4. Exact augmented class-specific order statistic.
5. Random class calibration count \(N_k\).
6. Conditional within-class exchangeability.
7. Treatment of ties.
8. Treatment of \(N_k=0\).
9. Treatment of unattainable finite ranks.
10. Interpretation of \(q_k=+\infty\).
11. Numeric-target inclusion implying ordinal-bin inclusion.
12. Proof that fallback and hull only add labels.
13. Conditional coverage given \(\mathcal F\).
14. Label-conditional coverage after averaging over \(\mathcal F\).
15. Discussion of target representation and monotone recoding.
16. Distinction between true numeric targets and surrogate class embeddings.
17. Limitations under chronological or distribution-shifted evaluation.

The full proposition--lemma--theorem structure should remain in this manuscript.

---

### 4.5 Experimental Scope

The journal-track manuscript should include all conference-track experiments plus substantial extensions.

Recommended extensions include:

#### Broader benchmark coverage

- additional ordinal medical datasets;
- additional facial age datasets;
- additional tabular ordinal datasets;
- additional rare-event or geoscience datasets, when scientifically justified.

#### Stronger baseline coverage

- additional ordinal conformal methods;
- RAPS if relevant;
- alternative CQR variants;
- class-conditional versions of standard CP baselines;
- regression-to-bin baselines;
- structured ordinal regression baselines.

#### Expanded ablations

- pooled versus class-specific calibration;
- raw versus hull prediction sets;
- fallback policy;
- alternative ordinal embeddings;
- alternative bin definitions;
- alternative base quantile levels;
- different calibration sizes;
- different target miscoverage levels;
- different backbone architectures;
- different score definitions;
- different split strategies.

#### Sensitivity and robustness

- class imbalance severity;
- calibration support per class;
- label noise;
- target--label inconsistencies;
- distribution shift;
- temporal extrapolation;
- repeated-group dependence;
- quantile crossing;
- nonfinite model outputs;
- small-sample behavior.

#### Efficiency and scalability

- training cost;
- calibration cost;
- test-time cost;
- memory use;
- complexity as a function of sample count and number of classes;
- runtime comparison with search-based or architecture-specific baselines.

---

### 4.6 Dataset Contracts and Reproducibility

The journal-track manuscript and repository should document, for every dataset:

- source fields for \(X\), \(Z\), and \(Y_{\mathrm{ord}}\);
- target transformation;
- ordinal thresholds;
- boundary convention;
- invalid or missing sample handling;
- retained-sample manifest;
- split procedure;
- split hash;
- target--label consistency validation;
- dataset contract version.

Calibration metadata should include:

- class identifier;
- \(N_k\);
- rank \(r_k\);
- correction \(q_k\);
- finite or infinite status;
- ties at the correction;
- score minimum and maximum;
- seed;
- checkpoint;
- code commit;
- configuration hash;
- dataset contract version.

The journal-track paper should discuss how these artifacts support reproducibility and prevent calibration or test leakage.

---

### 4.7 Results and Analysis Scope

The journal-track results should include:

- aggregate coverage and set size;
- per-class coverage;
- worst-class coverage;
- per-class set size;
- raw versus final prediction sets;
- empty-set frequency;
- fallback frequency;
- fragmented-set frequency;
- hull inflation;
- fallback inflation;
- full-set rate;
- calibration support and correction size;
- confidence intervals or variability across repeated runs;
- failure cases;
- qualitative examples where appropriate.

The paper should explain not only whether OCQR performs well, but also:

- when class-specific calibration becomes conservative;
- when the hull substantially increases set size;
- when surrogate embeddings affect efficiency;
- when exchangeability may be doubtful;
- when temporal evaluation should be interpreted empirically rather than as direct theorem validation.

---

### 4.8 Claims Allowed in the Journal Track

The journal-track paper may make broader claims, but only after full theoretical and empirical support.

Potential claims:

- a general class-conditional conformal framework for ordinal classification;
- a formal bridge from numeric conformal intervals to ordinal prediction sets;
- finite-sample class-conditional ordinal coverage under explicit assumptions;
- validity of add-only fallback and hull post-processing;
- representation-dependent efficiency for class-only datasets;
- broad empirical behavior across datasets, architectures, and imbalance regimes;
- reproducible implementation aligned with a formal method contract.

The paper should explicitly discuss limitations:

- exchangeability is assumed, not guaranteed by temporal splitting;
- class-specific validity may be inefficient with small \(N_k\);
- \(q_k=+\infty\) preserves validity but can produce uninformative sets;
- hull closure can increase prediction-set size;
- surrogate ordinal embeddings affect efficiency and membership;
- class-conditional coverage does not imply feature-conditional coverage;
- model-agnostic calibration does not correct poor base-model accuracy.

---

## 5. Content That May Appear in Both Manuscripts

Some overlap is scientifically unavoidable. The following may appear in both, but wording should be adapted and depth should differ:

- motivation for ordinal uncertainty quantification;
- definition of ordinal labels and bins;
- high-level OCQR workflow;
- nonconformity score;
- class-specific Mondrian calibration;
- candidate-specific corrections;
- bin-intersection rule;
- ordinal hull;
- main coverage statement;
- descriptions of shared datasets and baselines;
- selected main results.

Shared text should not be copied wholesale. The journal-track version should:

- expand the theory;
- add new experiments;
- add deeper analysis;
- update the literature review;
- revise wording and organization;
- clearly identify the earlier manuscript where required by publication policy.

---

## 6. Content Reserved Primarily for the Conference Track

The following should be treated as conference-track priorities:

- concise introduction of OCQR as a named method;
- focused comparison against LAC, APS, OAPS, and COPOC;
- four-dataset benchmark centered on Wine Quality, UTKFace, RetinaMNIST, and Solar Flare;
- compact theorem and proof sketch;
- core ablation of class-specific calibration and ordinal hull;
- emphasis on rare-class coverage and prediction-set efficiency;
- concise algorithm box.

These items may reappear in extended form in the journal track, but they define the focused identity of the conference-track paper.

---

## 7. Content Reserved Primarily for the Journal Track

The following should normally remain outside the conference main paper:

- full proposition and lemma sequence;
- complete conditioning on \(\mathcal F\);
- detailed handling of random \(N_k\);
- all \(+\infty\), tie, and empty-set edge cases;
- formal representation analysis;
- extensive dataset contracts;
- threshold dtype and boundary behavior;
- exhaustive deterministic tests;
- calibration metadata schema;
- implementation-to-contract mapping;
- configuration synchronization;
- broad sensitivity analysis;
- temporal extrapolation analysis;
- dependence and exchangeability limitations;
- extensive qualitative and failure analysis;
- full reproducibility audit.

---

## 8. AI Agent Routing Rules

When an AI agent receives a request to draft or revise content, it should apply the following decision process.

### Route to the conference track when the content:

- explains the core method briefly;
- supports a selected baseline;
- belongs to the four-dataset focused experiment;
- presents the main class-conditional coverage theorem concisely;
- reports primary coverage and set-size results;
- supports the central method contribution;
- can fit within a concise two-column paper.

### Route to the journal track when the content:

- adds formal assumptions or edge cases;
- introduces additional datasets or baselines;
- expands proofs;
- discusses implementation contracts;
- analyzes representation choices;
- reports robustness or sensitivity experiments;
- documents reproducibility metadata;
- studies temporal shift or dependence;
- explains failure modes or limitations in depth.

### Route to both, with different depth, when the content:

- defines the main problem;
- introduces OCQR;
- states the main guarantee;
- explains candidate-specific calibration;
- explains ordinal hull post-processing;
- reports the central benchmark result.

---

## 9. Terminology Rules

Use the following terms consistently:

- **ordinal label**: \(Y\) or \(Y_{\mathrm{ord}}\);
- **numeric target**: \(Z\);
- **ordinal bin**: \(B_k\);
- **class-specific conformal correction**: \(q_k\);
- **candidate-specific evaluation**: evaluating candidate \(k\) using \(q_k\);
- **raw prediction set**: \(C_{\mathrm{raw}}(x)\);
- **ordinal hull**: add all labels between the minimum and maximum selected labels;
- **full-label fallback**: return \(\mathcal Y\) when the raw set is empty;
- **class-conditional coverage** or **label-conditional coverage**;
- **within-class exchangeability**;
- **target--label consistency**.

Avoid:

- “class-specific quantile” when it may be confused with the fitted quantile-regression output;
- “ordinal closure” unless it is explicitly defined as the same operation as the ordinal hull;
- “conditional coverage” without specifying the conditioning variable;
- “continuous target” for class-index embeddings;
- “large-scale” without supporting evidence;
- “guaranteed under distribution shift”;
- “model-agnostic” if a claim concerns the quantile-regression head rather than the backbone.

---

## 10. Claim Verification Checklist

Before adding a claim to either manuscript, verify:

- [ ] Is the claim supported by the theory specification?
- [ ] Are all required assumptions stated nearby?
- [ ] Is the claim theoretical or empirical?
- [ ] Does the empirical metric directly support the claim?
- [ ] Is the comparison baseline implemented consistently?
- [ ] Does the claim apply to all datasets or only selected datasets?
- [ ] Does the claim depend on a specific target representation?
- [ ] Does the claim depend on finite class calibration support?
- [ ] Is a temporal experiment being described as empirical extrapolation rather than theorem validation?
- [ ] Is the wording narrower than the available evidence?

---

## 11. Experiment Allocation Matrix

| Experiment or Analysis | Conference Track | Journal Track |
|---|:---:|:---:|
| LAC comparison | Required | Required |
| APS comparison | Required | Required |
| OAPS comparison | Required | Required |
| COPOC comparison | Required | Required |
| Pooled/global calibration | Required | Required |
| OCQR without hull | Required | Required |
| Wine Quality | Required | Required |
| UTKFace | Required | Required |
| RetinaMNIST | Required | Required |
| Solar Flare | Required | Required |
| Additional datasets | Optional | Recommended |
| Alternative embeddings | Optional or omitted | Required ablation |
| Calibration-size sensitivity | Optional | Recommended |
| Miscoverage-level sensitivity | Limited | Recommended |
| Backbone sensitivity | Optional | Recommended |
| Temporal shift analysis | Brief | Detailed |
| Runtime and complexity | Brief if available | Detailed |
| Raw-empty and fallback analysis | Brief | Detailed |
| Infinite-correction analysis | Optional | Detailed |
| Label-noise robustness | Omitted | Optional or recommended |
| Target--label inconsistency analysis | Omitted | Recommended |
| Full reproducibility audit | Omitted | Required |

---

## 12. Recommended Repository Organization

```text
docs/
├── manuscript_scope.md
├── methods/
│   ├── ocqr_theory.md
│   └── ocqr_contract.md
├── manuscripts/
│   ├── conference_track/
│   │   ├── outline.md
│   │   ├── claims.md
│   │   ├── experiments.md
│   │   └── section_notes/
│   └── journal_track/
│       ├── outline.md
│       ├── claims.md
│       ├── experiments.md
│       └── section_notes/
└── datasets/
    ├── README.md
    ├── wine_quality.md
    ├── utkface.md
    ├── retinamnist.md
    └── solar_flare.md
```

Suggested location for this file:

```text
docs/manuscript_scope.md
```

---

## 13. Final Separation Principle

The conference-track manuscript should present a focused and complete first account of the core OCQR method.

The journal-track manuscript should not be a longer copy of that paper. It should provide a materially expanded contribution through:

- full theory;
- broader experiments;
- additional baselines;
- deeper ablation;
- robustness and sensitivity analysis;
- implementation validation;
- complete reproducibility documentation;
- a more general scientific framing.

When uncertain, place concise core content in the conference track and reserve formal depth, broader evidence, and implementation completeness for the journal track.
