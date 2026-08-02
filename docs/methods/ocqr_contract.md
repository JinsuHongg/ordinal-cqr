# OCQR Method Contract

**Status:** Target normative specification; not yet fully implemented
**Method:** Ordinal Conformalized Quantile Regression (OCQR)
**Version:** 0.3.0
**Companion theory:** `docs/methods/ocqr_theory.md`

This document is the normative implementation contract for OCQR. The code, tests, configuration, experiment outputs, and manuscript MUST agree with this document.

Dataset-specific numeric targets, transformations, ordinal labels, and bins MUST be defined in separate dataset contracts.

---

## 1. Canonical method

OCQR constructs an ordinal prediction set by:

1. fitting a common lower and upper quantile-regression predictor for a numeric target \(Z\);
2. correcting quantile crossing by deterministic endpoint sorting;
3. computing a CQR nonconformity score;
4. calibrating a separate Mondrian correction \(q_k\) using the supplied true ordinal label \(Y_{\mathrm{ord}}=k\);
5. evaluating candidate class \(k\) with its own correction \(q_k\);
6. including candidate \(k\) when its candidate-specific numeric acceptance set intersects bin \(B_k\);
7. applying the ordinal hull;
8. replacing an empty raw set with the full ordinal label space.

The target guarantee is finite-sample class-conditional ordinal coverage under the assumptions in `ocqr_theory.md`.

---

## 2. Data interface

Each observation MUST expose

\[
(X,Z,Y_{\mathrm{ord}}),
\]

where:

- \(X\in\mathcal X\) is the model input;
- \(Z\in\mathbb R\) is the numeric target used for pinball loss and the CQR score;
- \(Y_{\mathrm{ord}}\in\{0,\ldots,K-1\}\) is the ordinal label used for Mondrian grouping;
- \(B_k\subseteq\mathbb R\) is the fixed numeric bin associated with class \(k\).

The canonical implementation MUST accept \(Z\) and \(Y_{\mathrm{ord}}\) as logically distinct inputs, even when one is deterministically constructed from the other.

### 2.1 Required target-label consistency

Under the canonical bin contract, the bins form a disjoint partition of the supported target domain. Every retained observation MUST satisfy

\[
Y_{\mathrm{ord}}=k \Longrightarrow Z\in B_k.
\]

Because \(Y_{\mathrm{ord}}\) takes exactly one value and the canonical bins are disjoint and exhaustive, this requirement implies the equivalence

\[
Y_{\mathrm{ord}}=k \Longleftrightarrow Z\in B_k
\]

for retained canonical observations.

A dataset contract MAY state only the forward implication when using a noncanonical interface with overlapping bins, an incomplete supported domain, or ambiguous labels. Such an interface is outside the canonical bin contract and MUST be named as a separate method variant. The coverage theorem MAY still apply if the forward consistency implication and every other theorem assumption are established. Retained label-target inconsistencies that violate the forward implication are outside the theorem unless the prediction target is defined separately.

Calibration grouping MUST use the supplied \(Y_{\mathrm{ord}}\), not a class recomputed from \(Z\), unless the dataset contract proves the canonical equivalence and explicitly declares a derived-label interface.

### 2.2 Class-only datasets

When no observed numeric target exists, use a fixed documented embedding

\[
Z=e(Y_{\mathrm{ord}}).
\]

The canonical class-index embedding is

\[
e(k)=k.
\]

This is a surrogate ordinal coordinate, not a measured continuous quantity.

Validity holds for the fixed embedding and bins. Efficiency and candidate-set membership are generally **not invariant** to arbitrary monotone recodings of the ordinal labels. Alternative embeddings MUST therefore be named and reported as separate ablations.

### 2.3 Numeric-target datasets

When an observed numeric target exists, that target or a fixed deterministic transformation SHOULD be used as \(Z\). Examples include chronological age and transformed peak X-ray flux.

Continuous-target coverage implies true-bin inclusion under the consistency condition. The converse need not hold; ordinal coverage may therefore be more conservative than continuous-target interval coverage.

---

## 3. Bin contract

There MUST exist extended thresholds

\[
-\infty=b_0<b_1<\cdots<b_{K-1}<b_K=+\infty
\]

such that

\[
B_k=[b_k,b_{k+1}),\qquad k=0,\ldots,K-1.
\]

Thus the bins MUST be ordered, pairwise disjoint, and form a partition of the supported numeric target domain; canonical OCQR uses the whole real line as the supported domain.

For each internal threshold \(k=1,\ldots,K-1\), threshold equality belongs to the bin on the right:

\[
z=b_k\Longrightarrow z\in B_k.
\]

The extended endpoints \(b_0=-\infty\) and \(b_K=+\infty\) are not finite target values and are excluded from this equality convention.

Targets and thresholds MUST be promoted to a common floating-point dtype before bin comparisons. The implementation MUST NOT cast noninteger thresholds such as \(0.5\) to an integer target dtype.

For class-only embeddings, deterministic tests MUST verify every embedded class value and every threshold boundary using integer-valued input targets as well as floating-point inputs.

---

## 4. Base quantile regression

Let \(\alpha\in(0,1)\) be the target miscoverage level. Canonical experiments use

\[
\tau_L=\alpha/2,
\qquad
\tau_U=1-\alpha/2.
\]

These levels are modeling and efficiency choices, not requirements for split-conformal validity. Other fixed base quantile levels MAY be used in a separately configured variant if the same nonconformity score is calibrated exactly.

The model outputs

\[
\widehat q_{\tau_L}(x),\qquad \widehat q_{\tau_U}(x).
\]

### 4.1 Pinball loss

For \(\tau\in(0,1)\),

\[
\rho_\tau(u)=u\bigl(\tau-\mathbf 1\{u<0\}\bigr),
\qquad u=z-\widehat q_\tau(x).
\]

### 4.2 Crossing correction

Before calibration or prediction,

\[
\widehat\ell(x)=\min\{\widehat q_{\tau_L}(x),\widehat q_{\tau_U}(x)\},
\]

\[
\widehat u(x)=\max\{\widehat q_{\tau_L}(x),\widehat q_{\tau_U}(x)\}.
\]

All downstream operations MUST use \(\widehat\ell\) and \(\widehat u\).

### 4.3 Nonfinite outputs

Raw and corrected endpoints MUST be finite. If any endpoint is NaN or infinite, calibration or prediction MUST fail explicitly with an error. Nonfinite endpoints MUST NOT be converted into an empty raw set or a full prediction set.

---

## 5. CQR nonconformity score

Canonical OCQR uses the nonconformity score

\[
s(x,z)=\max\{\widehat\ell(x)-z,\;z-\widehat u(x)\}.
\]

Negative scores are allowed and MUST NOT be clipped.

For finite \(q\in\mathbb R\), define

\[
A_q(x)=\{z\in\mathbb R:s(x,z)\le q\}.
\]

If \(\widehat\ell(x),\widehat u(x),q\) are finite, then

\[
A_q(x)=
\begin{cases}
[\widehat\ell(x)-q,\widehat u(x)+q],
& \widehat\ell(x)-q\le \widehat u(x)+q,\\[4pt]
\varnothing,&\text{otherwise.}
\end{cases}
\]

For \(q=+\infty\), define \(A_q(x)=\mathbb R\).

The implementation MUST satisfy

\[
z\in A_q(x)\Longleftrightarrow s(x,z)\le q.
\]

---

## 6. True-label Mondrian calibration

Given calibration data

\[
\{(X_i,Z_i,Y_i)\}_{i=1}^{n},
\]

define

\[
\mathcal I_k=\{i:Y_i=k\},\qquad N_k=|\mathcal I_k|.
\]

Mondrian groups MUST be formed from the supplied true ordinal labels \(Y_i\).

For each \(i\in\mathcal I_k\), compute

\[
S_{k,i}=s(X_i,Z_i).
\]

Retain duplicate values and sort in nondecreasing order.

### 6.1 Exact augmented order statistic

Define

\[
r_k=\left\lceil(N_k+1)(1-\alpha)\right\rceil.
\]

Augment the class scores with one \(+\infty\) value:

\[
\widetilde{\mathcal S}_k
=
\{S_{k,i}:i\in\mathcal I_k\}\cup\{+\infty\}.
\]

Let \(\widetilde S_{k,(j)}\) denote the \(j\)-th order statistic of the augmented multiset. Define uniformly

\[
q_k=\widetilde S_{k,(r_k)}.
\]

This definition automatically gives \(q_k=+\infty\) when \(N_k=0\) or when the requested rank exceeds the number of finite calibration scores.

No interpolation and no randomized tie-breaking are used. Prediction uses \(\le q_k\).

### 6.2 Infinite correction interpretation

When \(q_k=+\infty\), candidate \(k\) is always included before hull closure. Consequently, its ordinal coverage is conservatively one. The limitation is not absence of a coverage guarantee; it is absence of an informative finite correction and useful efficiency guarantee.

---

## 7. Candidate-specific membership

For candidate class \(k\), define

\[
I_k(x)=A_{q_k}(x).
\]

The raw ordinal candidate set is

\[
C_{\mathrm{raw}}(x)
=
\{k:I_k(x)\cap B_k\neq\varnothing\}.
\]

Candidate \(k\) MUST be evaluated with \(q_k\), not with the correction of a point-predicted class.

For

\[
B_k=[b_k,b_{k+1})
\]

and a nonempty finite interval

\[
I_k(x)=[L_k(x),U_k(x)],
\]

membership is equivalent to

\[
U_k(x)\ge b_k
\quad\text{and}\quad
L_k(x)<b_{k+1}.
\]

---

## 8. Empty raw set and ordinal hull

Define the full ordinal label space as

\[
\mathcal Y=\{0,\ldots,K-1\}.
\]

The canonical pre-hull fallback is

\[
\widetilde C_{\mathrm{raw}}(x)=
\begin{cases}
C_{\mathrm{raw}}(x),&C_{\mathrm{raw}}(x)\neq\varnothing,\\[4pt]
\mathcal Y,&C_{\mathrm{raw}}(x)=\varnothing.
\end{cases}
\]

For any nonempty \(A\subseteq\mathcal Y\), define

\[
\operatorname{Hull}(A)
=
\{j\in\mathcal Y:\min A\le j\le\max A\}.
\]

The final set is

\[
C(x)=\operatorname{Hull}(\widetilde C_{\mathrm{raw}}(x)).
\]

Since \(\operatorname{Hull}(\mathcal Y)=\mathcal Y\), an empty raw set produces the full label space.

This fallback is conservative, nonempty, and contiguous. It preserves coverage because it only adds labels.

Experiments MUST record:

- raw empty-set rate;
- hull inflation relative to the original \(C_{\mathrm{raw}}\);
- fallback inflation;
- final full-set rate;
- fragmented raw-set rate.

---

## 9. Calibration and test independence

The fitted procedure and every method choice that can affect scores or prediction sets MUST be selected using only training and validation data that are independent of the calibration and test observations.

Before accessing calibration targets or labels, the following MUST be fixed:

- fitted model parameters and checkpoint;
- target transformation and numeric representation;
- ordinal bins and boundary convention;
- base quantile levels;
- nonconformity score;
- all hyperparameters and the canonical method variant;
- Mondrian taxonomy;
- empty-set fallback policy;
- hull policy;
- metric definitions used for model or method selection.

Calibration data MUST enter canonical OCQR only through the prespecified computation of the class corrections \(q_k\) and their required metadata. Calibration performance MUST NOT be used to choose among checkpoints, hyperparameters, embeddings, bins, scores, fallback policies, hull policies, or method variants.

Test data MUST remain untouched until final evaluation and MUST NOT be used to compute corrections or make any design choice.

---

## 10. Coverage claim

Under the assumptions in `ocqr_theory.md`, including conditional within-class exchangeability, an independently fitted and frozen procedure, target-bin consistency, exact augmented order statistics, true-label Mondrian grouping, candidate-specific inversion, and add-only post-processing, canonical OCQR claims the following for each class \(k\).

For almost every fitted procedure \(\mathcal F\),

\[
\Pr\{Y_{\mathrm{ord}}\in C(X)\mid \mathcal F,Y_{\mathrm{ord}}=k\}
\ge 1-\alpha.
\]

Averaging over the conditional distribution of \(\mathcal F\) given \(Y_{\mathrm{ord}}=k\) yields

\[
\Pr\{Y_{\mathrm{ord}}\in C(X)\mid Y_{\mathrm{ord}}=k\}
\ge 1-\alpha.
\]

For chronological solar-flare evaluation, this is an evaluation **under the theorem's exchangeability assumption**, not proof that the assumption holds. Empirical coverage under temporal extrapolation MUST be reported separately.

---

## 11. Required calibration metadata

For every class and run, store:

- class identifier;
- \(N_k\);
- requested rank \(r_k\);
- whether \(q_k\) is finite;
- \(q_k\), preserving `+inf` explicitly;
- count of calibration scores tied at finite \(q_k\);
- minimum and maximum finite calibration score;
- split hash;
- configuration hash;
- code commit;
- random seed;
- checkpoint identifier;
- target/bin contract version.

Legacy outputs lacking the corrected metric schema or method version MUST NOT be mixed into canonical tables.

---

## 12. Required deterministic tests

Tests MUST cover:

1. all integer embedded targets with midpoint thresholds;
2. dtype promotion of integer targets and floating thresholds;
3. equality at every included and excluded bin boundary;
4. raw quantile crossing;
5. NaN and infinite endpoints raising explicit errors;
6. negative nonconformity scores;
7. negative finite corrections;
8. empty numeric acceptance sets;
9. tied calibration scores;
10. exact augmented order statistic;
11. \(N_k=0\);
12. unattainable finite rank producing \(+\infty\);
13. true-label grouping distinct from target-derived grouping;
14. score inversion;
15. candidate-specific corrections;
16. nonempty bin intersection;
17. fragmented raw sets;
18. ordinal hull inclusion and contiguity;
19. empty raw set producing the full label space;
20. raw-empty, hull-inflation, fallback-inflation, and full-set metrics;
21. target-label-bin consistency for every dataset.

Repeated synthetic experiments MUST validate per-class coverage under simulated within-class exchangeability, including random class counts.

---

## 13. Dataset-specific obligations

Each dataset contract MUST specify:

- source fields for \(Z\) and \(Y_{\mathrm{ord}}\);
- transformation of \(Z\);
- all thresholds including \(\pm\infty\);
- boundary convention;
- missing/invalid handling;
- automated target-label-bin consistency tests;
- representation limitations for class-only embeddings.

---

## 14. Current implementation mapping

| Contract component | Current implementation | Tests/status |
|---|---|---|
| Bin/class indexing | `src/ocqr_solar/explainability/poshoc_uc.py::OrdinalCQRWrapper._class_indices` | MUST fix integer-threshold dtype casting and add integer-target boundary tests |
| Empty raw-set fallback | `src/ocqr_solar/explainability/poshoc_uc.py::OrdinalCQRWrapper._ordinal_hull` | Implemented as the canonical v0.3 full-label fallback; existing fallback test covers it |
| Mondrian grouping | `src/ocqr_solar/explainability/poshoc_uc.py::OrdinalCQRWrapper.calibrate`; `OrdinalCQRWrapper.predict_step` | MUST accept logically distinct \(Z\) and supplied \(Y_{\mathrm{ord}}\); distinct-label test is missing |
| Existing ordinal CQR behavior | `tests/test_ordinal_cqr.py` | Update and expand contract tests |
| Pinball loss | `src/ocqr_solar/utils/losses.py::PinballLoss.forward`; wired by `src/ocqr_solar/models/module.py::ResNetQR.__init__` | Implemented; focused formula and configured-quantile contract tests are missing |
| Crossing correction | `src/ocqr_solar/explainability/poshoc_uc.py::OrdinalCQRWrapper.calibrate`; `OrdinalCQRWrapper.predict_step` | Implemented with endpoint sorting and covered by the existing crossing test |
| Nonfinite endpoint rejection | No explicit check in `OrdinalCQRWrapper.calibrate` or `OrdinalCQRWrapper.predict_step` | MUST implement explicit calibration/prediction errors and tests |
| Calibration metadata | `TODO: repository audit` | Not yet complete |
| Hull/fallback metrics | `TODO: repository audit` | Not yet complete |

Line numbers are review-time references and MUST be replaced by stable function names during the implementation audit.


## 15. Configuration synchronization

The canonical method configuration MUST declare the same method version and policies as this contract. At minimum:

```yaml
method:
  name: ocqr
  version: "0.3.0"

prediction:
  empty_candidate_set: full_label_space
```

A configuration that declares `preserve_empty` or an earlier method version MUST NOT be used for canonical v0.3 results.
