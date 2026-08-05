# Benchmark summaries

The CSV files in this directory contain final prediction-set metrics computed
from the saved per-sample test outputs at \(\alpha=0.1\). `CCR` is contiguous
coverage rate: a set must contain the true label and have exactly one
contiguous segment. Every reported final set in these two runs was contiguous,
so CCR equals marginal coverage.

`raw_fragmentation_rate` and `avg_hull_inflation` apply only to OCQR, whose
saved output includes a raw candidate set before the ordinal hull. `NA` means
the method did not expose an analogous intermediate set. The RetinaMNIST CQR
row is a diagnostic: its continuous intervals were mapped to the same ordinal
bins after prediction, rather than produced as native set predictions.

For paper reporting, also track worst-class coverage (with counts and binomial
confidence intervals), average set size, full-set rate, SFS/MDJ fragmentation,
per-class calibration support and corrections, and artifacts that identify the
exact split manifest, configuration, checkpoint, code commit, and random seed.
The current UTKFace split is derived from an unsorted filename listing and is
therefore exploratory until its filename manifests are frozen.

`all_datasets_uq_summary_alpha_0.1.csv` is the advisor-facing comparison table.
Its `empirical_*_target_met` columns compare point estimates with the nominal
0.9 target; they do not establish a conformal theorem assumption or prove
population validity. `claim_scope=class_conditional_under_exchangeability`
identifies OCQR runs whose method-level guarantee applies only under the
documented exchangeability, frozen-procedure, and dataset-contract assumptions.
The EyePACS rows are exploratory because its split is not patient/eye-disjoint
and the test set has been inspected during model development.

`all_datasets_uq_per_class_alpha_0.1.csv` is the normalized companion table.
It includes class sample counts, point coverage, average set size, and Wilson
95% confidence intervals. The intervals quantify test-sample uncertainty; they
do not replace the conformal assumptions or turn an empirical result into a
theoretical guarantee.
