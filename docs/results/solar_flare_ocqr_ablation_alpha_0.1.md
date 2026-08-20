# Solar Flare OCQR post-processing ablations (alpha = 0.1)

Five fixed QR checkpoints (seeds 0–4) are evaluated on the same 27,620 image-available future-test examples. Values are mean ± sample standard deviation across seeds.

| Variant | Status | Marginal coverage | Macro coverage | Worst-class coverage | Mean set size | Clipped corrections |
|---|---|---:|---:|---:|---:|---:|
| OCQR | Canonical | 89.64% ± 5.44% | 88.35% ± 6.16% | 72.97% ± 17.76% | 2.60 ± 0.11 | -- |
| OCQR-Pooled | Noncanonical pooled ablation | 88.02% ± 6.05% | 80.76% ± 9.06% | 43.92% ± 24.99% | 2.42 ± 0.13 | 0.00% ± 0.00% |
| OCQR-NoHull | Noncanonical diagnostic | 89.64% ± 5.44% | 88.35% ± 6.16% | 72.97% ± 17.76% | 2.60 ± 0.11 | -- |
| OCQR-NoFallback | Noncanonical diagnostic | 89.64% ± 5.44% | 88.35% ± 6.16% | 72.97% ± 17.76% | 2.60 ± 0.11 | -- |
| OCQR-Raw | Noncanonical diagnostic | 89.64% ± 5.44% | 88.35% ± 6.16% | 72.97% ± 17.76% | 2.60 ± 0.11 | -- |
| OCQR-NonnegativeCorrection | Exploratory post hoc | 92.55% ± 3.46% | 91.92% ± 3.70% | 85.64% ± 6.65% | 2.83 ± 0.04 | 48.00% ± 22.80% |

`OCQR-Pooled` uses one correction estimated from all calibration examples rather than true-label Mondrian corrections. It is a noncanonical ablation that isolates class-specific calibration.

`OCQR-NonnegativeCorrection` clips each class correction at zero. It was introduced after inspecting low future-test B-class coverage and is consequently an exploratory post-hoc robustness diagnostic, not canonical OCQR or confirmatory evidence.

The companion CSV records all aggregate diagnostics and per-class coverage. The no-hull, no-fallback, and raw variants are identical to canonical OCQR in these runs because no raw set was empty or fragmented.
