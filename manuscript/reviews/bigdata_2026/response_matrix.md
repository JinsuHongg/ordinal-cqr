# BigData 2026 reviewer response matrix

Keep each reviewer comment independently traceable. Cross-references identify overlapping work; they do not replace the original comment IDs. Status refers to manuscript or experiment work, not to whether the comment has been acknowledged.

**Source:** [`reviews_original.txt`](reviews_original.txt)  
**Working manuscript:** [`../../main.tex`](../../main.tex)  
**Overall status:** Responses and revisions have not yet been drafted.

## Reviewer 1

| ID | Issue | Planned response / work | Manuscript location | Evidence or experiment | Status |
|---|---|---|---|---|---|
| R1-M1 | Realism and scope of within-class exchangeability and target-label consistency | Explain what each assumption means in the two data regimes, when it is plausible, and what violations imply. Link to R1-M2 for temporal shift. | Proposed Methodology; Limitations | `docs/methods/ocqr_theory.md`; dataset contracts in `docs/datasets/` | Not started |
| R1-M2 | Solar future-test results do not verify exchangeability | Check the abstract, results, table captions, and conclusion for consistent language: these are temporal-extrapolation results, not confirmation of the guarantee. | Abstract; Results and Discussion; Limitations; Conclusion | `outputs/conference_v0_3/solar_flare/`; solar split and manifest audit artifacts | Not started |
| R1-M3 | Backbone differences confound method comparisons | Assess a controlled comparison with the same backbone/predictions and conformal scores. Connect to R2-D3. | Baselines and Ablations; Results and Discussion; Limitations | Current per-method outputs in `outputs/conference_v0_3/`; additional controlled runs may be needed | Not started |
| R1-M4 | Only 90% nominal coverage evaluated | Plan a coverage-versus-set-size sweep at 80%, 90%, and 95%, including per-class behavior. Connect to R2-D4. | Evaluation Metrics; Results and Discussion | New runs and an aggregation artifact are needed | Not started |
| R1-M5 | Rare-class estimates are uncertain | Report uncertainty intervals for rare-class coverage and qualify conclusions using the test counts. Connect to R1-m4. | Per-class Results; Evaluation Metrics; Limitations | `results/conference_v0_3/per_class_results.csv`; per-run prediction artifacts under `outputs/conference_v0_3/` | Not started |
| R1-M6 | Numeric ordinal encoding may affect OCQR | Evaluate reasonable alternative encodings for the class-only RetinaMNIST representation, or justify the chosen embedding and boundaries. | Problem Formulation; Datasets; Results and Discussion | New RetinaMNIST sensitivity runs may be needed | Not started |
| R1-M7 | Contributions of calibration, hull, and fallback need separation | Make the distinct statistical and structural roles explicit and check that the ablation interpretation matches saved runs. | Proposed Methodology; Ablation Analysis | `results/conference_v0_3/ablation_results.csv`; solar summary at `docs/results/solar_flare_ocqr_ablation_alpha_0.1.csv`; seed-level metrics under `outputs/conference_v0_3/` | Not started |
| R1-m1 | Calibration-set-size sensitivity | Assess a calibration-size sweep and its effect on coverage, set size, and class support. | Results and Discussion; Limitations | New runs or a controlled resampling analysis are needed | Not started |
| R1-m2 | Computational overhead | Report calibration and inference overhead with the measurement protocol and hardware context. | Implementation Details; Results and Discussion | Per-run timing fields are present in some `*_evaluation_metrics.json` artifacts; verify coverage across datasets and methods | Not started |
| R1-m3 | Distribution of prediction-set sizes | Add a distribution summary (for example, size frequencies or quantiles) alongside mean set size. | Evaluation Metrics; Results and Discussion | Per-example test-set predictions under `outputs/conference_v0_3/` | Not started |
| R1-m4 | Uncertainty intervals for rare classes | Address together with R1-M5 while retaining this minor-comment ID in the response. | Per-class Results; Evaluation Metrics | Same evidence as R1-M5 | Not started |

## Reviewer 2

| ID | Issue | Planned response / work | Manuscript location | Evidence or experiment | Status |
|---|---|---|---|---|---|
| R2-D1 | Novelty relative to standard Mondrian split conformal; numeric embedding may reduce to ordinal-index regression on RetinaMNIST | Sharpen the contribution statement, distinguish standard conformal ingredients from the OCQR construction, and state the limits of the class-index target representation. | Introduction; Proposed Methodology; Conclusion | `docs/methods/ocqr_theory.md`; `docs/methods/ocqr_contract.md` | Not started |
| R2-D2 | Baselines may not match the class-conditional claim; requests Mondrian baselines and clustered conformal | Audit which current baselines are pooled versus class-conditional. Decide whether to add the requested comparisons or narrow the comparison claims; do not describe existing marginal baselines as class-conditional. | Baselines and Ablations; Results and Discussion | Current run configs and provenance under `outputs/conference_v0_3/`; additional baseline runs may be needed | Not started |
| R2-D3 | Different backbones confound comparison | Handle jointly with R1-M3 and report what the controlled comparison does and does not isolate. | Baselines and Ablations; Limitations | Same evidence as R1-M3 | Not started |
| R2-D4 | High coverage can produce uninformative large sets; requests a coverage-size curve | Address jointly with R1-M4. Include RetinaMNIST operating points and discuss whether the set sizes are practically useful. | Overall Performance; Coverage-Set Size Tradeoff | New alpha-sweep results are needed; current aggregate is in `results/conference_v0_3/main_results.csv` | Not started |

## Reviewer 3

| ID | Comment | Handling | Manuscript location | Status |
|---|---|---|---|
| R3-GEN | Supportive comment on ordinal contiguity and possible clinical relevance | Preserve as positive feedback. Do not treat the suggested clinical relevance as empirical evidence or add clinical-effectiveness claims without supporting data. | No required change identified | Recorded |

## Response drafting checklist

- Keep the original reviewer wording in `reviews_original.txt`; do not edit it while revising the response.
- Update each status to `In progress`, `Addressed`, `Partially addressed`, or `Declined with rationale` as work proceeds.
- Before submission, replace planned work with the actual change, cite the revised section/table/figure, and link any new run configuration and result artifact.
- If a requested experiment is not run, state that clearly in the response and adjust the manuscript claim where needed.
