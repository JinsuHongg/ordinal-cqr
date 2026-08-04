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
