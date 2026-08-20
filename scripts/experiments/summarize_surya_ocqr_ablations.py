#!/usr/bin/env python3
"""Aggregate five-seed Solar Flare OCQR post-processing ablations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

VARIANTS = (
    ("ordinal_cqr", "OCQR", "Canonical"),
    ("ocqr_pooled", "OCQR-Pooled", "Noncanonical pooled ablation"),
    ("ocqr_no_hull", "OCQR-NoHull", "Noncanonical diagnostic"),
    ("ocqr_no_fallback", "OCQR-NoFallback", "Noncanonical diagnostic"),
    ("ocqr_raw", "OCQR-Raw", "Noncanonical diagnostic"),
    (
        "ocqr_nonnegative_correction",
        "OCQR-NonnegativeCorrection",
        "Exploratory post hoc",
    ),
)
METRICS = (
    "marginal_coverage",
    "macro_class_coverage",
    "worst_class_coverage",
    "avg_set_size",
    "ccr",
    "avg_sfs",
    "avg_mdj",
    "raw_empty_rate",
)


def one(path: Path) -> dict:
    return json.loads(path.read_text())


def summary(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for directory, label, status in VARIANTS:
        evaluations = []
        metadata = []
        for seed in range(5):
            result_dir = args.root / f"seed_{seed}" / "posthoc" / directory
            evaluation = list(result_dir.glob("*_evaluation_metrics.json"))
            calibration = list(result_dir.glob("*_calibration_metadata.json"))
            if len(evaluation) != 1 or len(calibration) != 1:
                raise RuntimeError(f"seed {seed}/{directory}: expected one evaluation and calibration JSON")
            evaluations.append(one(evaluation[0]))
            metadata.append(one(calibration[0]))

        counts = {entry["count"] for result in evaluations for entry in result["per_class"]}
        if len({result["num_samples"] for result in evaluations}) != 1:
            raise RuntimeError(f"{directory}: test sample count differs across seeds")
        row: dict[str, object] = {
            "variant": label,
            "experimental_status": status,
            "n_seeds": len(evaluations),
            "num_test": evaluations[0]["num_samples"],
        }
        for metric in METRICS:
            if metric == "macro_class_coverage":
                values = [statistics.mean(entry["coverage"] for entry in result["per_class"]) for result in evaluations]
            elif metric == "worst_class_coverage":
                values = [min(entry["coverage"] for entry in result["per_class"]) for result in evaluations]
            else:
                values = [result["aggregate"][metric] for result in evaluations]
            mean, std = summary(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        fractions = [item.get("postprocessing", {}).get("clipped_correction_fraction") for item in metadata]
        if all(value is not None for value in fractions):
            row["clipped_correction_fraction_mean"], row["clipped_correction_fraction_std"] = summary(fractions)
        else:
            row["clipped_correction_fraction_mean"] = "NA"
            row["clipped_correction_fraction_std"] = "NA"
        aggregate_rows.append(row)

        for class_id in range(5):
            entries = [result["per_class"][class_id] for result in evaluations]
            coverage_mean, coverage_std = summary([entry["coverage"] for entry in entries])
            class_rows.append({
                "variant": label,
                "experimental_status": status,
                "class_id": class_id,
                "class_name": entries[0]["class_name"],
                "test_count": entries[0]["count"],
                "coverage_mean": coverage_mean,
                "coverage_std": coverage_std,
            })

    fields = list(aggregate_rows[0])
    with (args.output_dir / "solar_flare_ocqr_ablation_alpha_0.1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (args.output_dir / "solar_flare_ocqr_ablation_per_class_alpha_0.1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(class_rows)

    def percent(mean: float, std: float) -> str:
        return f"{100 * mean:.2f}% ± {100 * std:.2f}%"
    def decimal(mean: float, std: float) -> str:
        return f"{mean:.2f} ± {std:.2f}"
    lines = [
        "# Solar Flare OCQR post-processing ablations (alpha = 0.1)",
        "",
        "Five fixed QR checkpoints (seeds 0–4) are evaluated on the same 27,620 image-available future-test examples. Values are mean ± sample standard deviation across seeds.",
        "",
        "| Variant | Status | Marginal coverage | Macro coverage | Worst-class coverage | Mean set size | Clipped corrections |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        clip = row["clipped_correction_fraction_mean"]
        clipped = "--" if clip == "NA" else percent(clip, row["clipped_correction_fraction_std"])
        lines.append(
            f"| {row['variant']} | {row['experimental_status']} | "
            f"{percent(row['marginal_coverage_mean'], row['marginal_coverage_std'])} | "
            f"{percent(row['macro_class_coverage_mean'], row['macro_class_coverage_std'])} | "
            f"{percent(row['worst_class_coverage_mean'], row['worst_class_coverage_std'])} | "
            f"{decimal(row['avg_set_size_mean'], row['avg_set_size_std'])} | {clipped} |")
    lines += [
        "",
        "`OCQR-Pooled` uses one correction estimated from all calibration examples rather than true-label Mondrian corrections. It is a noncanonical ablation that isolates class-specific calibration.",
        "",
        "`OCQR-NonnegativeCorrection` clips each class correction at zero. It was introduced after inspecting low future-test B-class coverage and is consequently an exploratory post-hoc robustness diagnostic, not canonical OCQR or confirmatory evidence.",
        "",
        "The companion CSV records all aggregate diagnostics and per-class coverage. The no-hull, no-fallback, and raw variants are identical to canonical OCQR in these runs because no raw set was empty or fragmented.",
        "",
    ]
    (args.output_dir / "solar_flare_ocqr_ablation_alpha_0.1.md").write_text("\n".join(lines))

if __name__ == "__main__":
    main()
