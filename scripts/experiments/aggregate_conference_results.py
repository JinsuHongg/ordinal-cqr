"""Validate and aggregate frozen conference-v0.3 run artifacts.

The aggregator is intentionally strict: a malformed or incomparable run is a
hard error, never a row that is silently mixed into a manuscript table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ordinal_cqr.experiments.prediction_artifacts import evaluate, load_predictions


METHOD_VERSION = "0.3.0"
SEEDS = {0, 1, 2, 3, 4}
PRIMARY_METRICS = (
    "marginal_coverage",
    "macro_class_coverage",
    "worst_class_coverage",
    "mean_set_size",
    "full_set_rate",
)
ABLATION_METRICS = (*PRIMARY_METRICS, "ccr", "avg_sfs", "avg_mdj", "raw_empty_rate")
METHOD_ORDER = ("lac", "aps", "oaps", "copoc", "ocqr")
METHOD_LABELS = {
    "lac": "LAC",
    "aps": "APS",
    "oaps": "OAPS",
    "copoc": "COPOC",
    "ocqr": "OCQR",
}
ABLATION_ORDER = ("ocqr_pooled", "ocqr_no_hull", "ocqr_no_fallback", "ocqr_raw", "ocqr_nonnegative_correction", "ocqr")
ABLATION_LABELS = {
    "ocqr_pooled": "OCQR-Pooled",
    "ocqr_no_hull": "OCQR-NoHull",
    "ocqr_no_fallback": "OCQR-NoFallback",
    "ocqr_raw": "OCQR-Raw",
    "ocqr_nonnegative_correction": "OCQR-NonnegativeCorrection",
    "ocqr": "OCQR",
}
DATASET_LABELS = {"retinamnist": "RetinaMNIST", "utkface": "UTKFace"}
REQUIRED_PROVENANCE = {
    "dataset", "dataset_contract_version", "method", "method_version", "alpha",
    "seed", "split_identifier", "split_hash", "configuration_hash", "protocol_hash", "code_commit",
    "checkpoint_identifier", "training_criterion", "checkpoint_selection_criterion",
    "runtime_seconds", "hardware", "timestamp",
}
REQUIRED_PREDICTION_FIELDS = {
    "sample_id", "Y_ord", "Z", "point_prediction", "prediction_set_raw",
    "prediction_set_final",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{description} must be a finite numeric value.")


def _validate_infinity_encoding(value: Any, path: Path) -> None:
    """Permit the canonical string +inf, never JSON nonfinite values or aliases."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: nonfinite JSON number is forbidden; use '+inf'.")
    if isinstance(value, str) and "inf" in value.lower() and value != "+inf":
        raise ValueError(f"{path}: infinity must be encoded exactly as '+inf'.")
    if isinstance(value, dict):
        for child in value.values():
            _validate_infinity_encoding(child, path)
    elif isinstance(value, list):
        for child in value:
            _validate_infinity_encoding(child, path)


def _prediction_path(run: Path) -> Path:
    matches = [run / f"predictions{suffix}" for suffix in (".jsonl", ".csv", ".parquet")
               if (run / f"predictions{suffix}").is_file()]
    if len(matches) != 1:
        raise ValueError(f"{run}: exactly one predictions artifact is required.")
    if matches[0].suffix != ".jsonl":
        raise ValueError(f"{run}: only schema-validated predictions.jsonl is currently supported.")
    return matches[0]


def _validate_predictions(path: Path, num_classes: int) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError(f"{path}: predictions cannot be empty.")
    for line_number, line in enumerate(rows, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSONL prediction.") from error
        if not isinstance(row, dict) or REQUIRED_PREDICTION_FIELDS - row.keys():
            raise ValueError(f"{path}:{line_number}: missing required prediction fields.")
        if row["sample_id"] in seen or not isinstance(row["Y_ord"], int):
            raise ValueError(f"{path}:{line_number}: duplicate sample ID or invalid ordinal label.")
        if not 0 <= row["Y_ord"] < num_classes:
            raise ValueError(f"{path}:{line_number}: ordinal label is out of range.")
        if not isinstance(row["point_prediction"], int) or not 0 <= row["point_prediction"] < num_classes:
            raise ValueError(f"{path}:{line_number}: point prediction is out of range.")
        _finite_number(row["Z"], f"{path}:{line_number}: numeric target")
        seen.add(row["sample_id"])
        for key in ("prediction_set_raw", "prediction_set_final"):
            value = row[key]
            if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
                raise ValueError(f"{path}:{line_number}: invalid {key}.")
            if value != sorted(set(value)) or any(not 0 <= item < num_classes for item in value):
                raise ValueError(f"{path}:{line_number}: {key} must be sorted, unique, and in range.")
        _validate_infinity_encoding(row, path)
    return [json.loads(line) for line in rows]


def _validate_run(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = run / "config.yaml"
    if not config.is_file():
        raise ValueError(f"{run}: resolved config.yaml is required.")
    provenance, metrics = (_read_json(run / name) for name in ("provenance.json", "metrics.json"))
    missing = REQUIRED_PROVENANCE - provenance.keys()
    if missing:
        raise ValueError(f"{run}: missing provenance fields {sorted(missing)}")
    if provenance["method_version"] != METHOD_VERSION:
        raise ValueError(f"{run}: method version is not canonical {METHOD_VERSION}")
    status = _read_json(run / "run_status.json")
    if status.get("status") != "evaluation_complete" or status.get("stage") != "complete":
        raise ValueError(f"{run}: run status is not evaluation_complete/complete.")
    if status.get("aggregation_eligible") is False:
        raise ValueError(f"{run}: run is explicitly aggregation-ineligible.")
    if not isinstance(provenance["code_commit"], str) or re.fullmatch(
        r"[0-9a-f]{40}", provenance["code_commit"]
    ) is None:
        raise ValueError(f"{run}: code_commit must be an exact 40-character Git SHA.")
    if provenance.get("git_dirty") is True:
        raise ValueError(f"{run}: canonical aggregation rejects a dirty source tree.")
    if provenance["method"] == "lac":
        required_lac = {
            "lac_method_version": "1.0.0-exact-split",
            "score": "one_minus_true_class_probability",
            "calibration": "pooled_exact_augmented_rank",
            "prediction_rule": "probability_superlevel_set",
            "inclusion": "non_strict",
        }
        if provenance.get("lac") != required_lac:
            raise ValueError(f"{run}: lac provenance is not canonical exact split LAC.")
    if provenance["method"] == "aps":
        required_aps = {
            "aps_method_version": "1.0.0-nonrandomized-boundary",
            "score": "cumulative_probability_through_true_label",
            "calibration": "pooled_exact_augmented_rank",
            "prediction_rule": "smallest_stable_probability_prefix_reaching_q",
            "probability_tie_rule": "ascending_class_index",
        }
        if provenance.get("aps") != required_aps:
            raise ValueError(f"{run}: aps provenance is not the canonical boundary rule.")
    if provenance["method"] == "copoc":
        required_copoc = {
            "copoc_method_version": "1.0.0-eq5-aps",
            "model_type": "resnet18_copoc_nonparametric_eq5",
            "phi": "abs",
            "psi_even": "negative_abs",
            "conformal_procedure": "aps",
            "checkpoint_selection_metric": "validation_cross_entropy",
        }
        if provenance.get("copoc") != required_copoc:
            raise ValueError(f"{run}: copoc provenance is not canonical Eq. (5) + APS.")
    if provenance["method"] == "oaps":
        required_oaps = {
            "oaps_method_version": "1.0.0-lu2022-algorithm1",
            "set_family": "greedy_mode_centered_adjacent_expansion",
            "calibration": "pooled_exact_augmented_rank",
            "mode_tie_rule": "lowest_class",
            "adjacent_tie_rule": "upper_right",
        }
        if provenance.get("oaps") != required_oaps:
            raise ValueError(f"{run}: oaps provenance is not canonical Lu et al. Algorithm 1.")
    if provenance["configuration_hash"] != _sha256(config):
        raise ValueError(f"{run}: configuration_hash does not match config.yaml")
    if provenance["seed"] not in SEEDS:
        raise ValueError(f"{run}: seed must be one of {sorted(SEEDS)}")
    _finite_number(provenance["alpha"], f"{run}: alpha")
    _finite_number(provenance["runtime_seconds"], f"{run}: runtime_seconds")
    calibration_path = run / "calibration.json"
    if not calibration_path.is_file():
        raise ValueError(f"{run}: calibration.json is required.")
    _validate_infinity_encoding(_read_json(calibration_path), calibration_path)
    manifest_reference = _read_json(run / "manifest_reference.json")
    if manifest_reference.get("manifest_sha256") != provenance["split_hash"]:
        raise ValueError(f"{run}: manifest reference and provenance split hash disagree.")
    aggregate = metrics.get("aggregate")
    per_class = metrics.get("per_class")
    if not isinstance(aggregate, dict) or not isinstance(per_class, list) or not per_class:
        raise ValueError(f"{run}: metrics must provide nonempty aggregate and per_class values.")
    for metric in PRIMARY_METRICS:
        if metric not in aggregate:
            raise ValueError(f"{run}: missing primary metric {metric}")
        _finite_number(aggregate[metric], f"{run}: {metric}")
    for entry in per_class:
        if not isinstance(entry, dict) or {"class_id", "count", "coverage"} - entry.keys():
            raise ValueError(f"{run}: per_class rows require class_id, count, and coverage.")
        _finite_number(entry["coverage"], f"{run}: per-class coverage")
        if not isinstance(entry["count"], int) or entry["count"] < 1:
            raise ValueError(f"{run}: per-class count must be a positive integer.")
    class_ids = [entry["class_id"] for entry in per_class]
    if class_ids != list(range(len(per_class))):
        raise ValueError(f"{run}: per-class metrics must cover consecutive ordinal labels.")
    predictions = _validate_predictions(_prediction_path(run), len(per_class))
    expected_test_count = manifest_reference.get("split_counts", {}).get("test")
    if expected_test_count != len(predictions):
        raise ValueError(f"{run}: prediction count does not match the manifest test count.")
    observed_counts = [0] * len(per_class)
    for prediction in predictions:
        observed_counts[prediction["Y_ord"]] += 1
    if observed_counts != [entry["count"] for entry in per_class]:
        raise ValueError(f"{run}: prediction labels and per-class test counts disagree.")
    if provenance["method"].startswith("ocqr_") or provenance["method"] == "ocqr":
        # Older canonical OCQR runs predate structural ablation metrics, but
        # retain the complete prediction artifact needed to recompute them.
        # Preserve recorded primary metrics and fill only missing diagnostics.
        recomputed = evaluate(
            load_predictions(_prediction_path(run), len(per_class)),
            len(per_class),
            float(provenance["alpha"]),
            ocqr=True,
        )["aggregate"]
        for metric in ABLATION_METRICS:
            aggregate.setdefault(metric, recomputed[metric])
    return provenance, metrics


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{" + "l" * len(fields) + "}\n\\toprule\n")
        stream.write(" & ".join(field.replace("_", "\\_") for field in fields) + " \\\\\n\\midrule\n")
        for row in rows:
            stream.write(" & ".join(str(row.get(field, "")).replace("_", "\\_") for field in fields) + " \\\\\n")
        stream.write("\\bottomrule\n\\end{tabular}\n")


def _write_json(path: Path, payload: object) -> None:
    """Write strict, deterministically ordered JSON result summaries."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def _dataset_sort_key(dataset: str) -> tuple[int, str]:
    return ({"retinamnist": 0, "utkface": 1}.get(dataset, 99), dataset)


def _method_sort_key(method: str) -> tuple[int, str]:
    return (
        METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER),
        method,
    )


def _format_mean_std(mean_value: float, std_value: float, *, percent: bool) -> str:
    """Format a repeated-run scalar for a manuscript LaTeX table."""
    if percent:
        return f"{100 * mean_value:.1f} $\\pm$ {100 * std_value:.1f}"
    return f"{mean_value:.2f} $\\pm$ {std_value:.2f}"


def _main_table_rows(summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in sorted(
        summaries,
        key=lambda row: (_dataset_sort_key(row["dataset"]), _method_sort_key(row["method"])),
    ):
        rows.append(
            {
                "dataset": DATASET_LABELS.get(summary["dataset"], summary["dataset"]),
                "method": METHOD_LABELS.get(summary["method"], summary["method"].upper()),
                "marginal_coverage": _format_mean_std(
                    summary["marginal_coverage_mean"],
                    summary["marginal_coverage_std"],
                    percent=True,
                ),
                "macro_class_coverage": _format_mean_std(
                    summary["macro_class_coverage_mean"],
                    summary["macro_class_coverage_std"],
                    percent=True,
                ),
                "worst_class_coverage": _format_mean_std(
                    summary["worst_class_coverage_mean"],
                    summary["worst_class_coverage_std"],
                    percent=True,
                ),
                "mean_set_size": _format_mean_std(
                    summary["mean_set_size_mean"],
                    summary["mean_set_size_std"],
                    percent=False,
                ),
                "full_set_rate": _format_mean_std(
                    summary["full_set_rate_mean"],
                    summary["full_set_rate_std"],
                    percent=True,
                ),
            }
        )
    return rows


def _summarize_per_class(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize per-class coverage over the five end-to-end seed runs."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], row["class_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, method, class_id), group in sorted(
        grouped.items(), key=lambda item: (_dataset_sort_key(item[0][0]), item[0][2], _method_sort_key(item[0][1]))
    ):
        if {row["seed"] for row in group} != SEEDS or len(group) != len(SEEDS):
            raise ValueError(
                f"{dataset}/{method}/class_{class_id}: exactly one five-seed per-class result is required."
            )
        counts = {row["count"] for row in group}
        if len(counts) != 1:
            raise ValueError(
                f"{dataset}/{method}/class_{class_id}: test class count changes across seeds."
            )
        coverages = [row["coverage"] for row in group]
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "class_id": class_id,
                "test_count": counts.pop(),
                "coverage_mean": statistics.mean(coverages),
                "coverage_std": statistics.stdev(coverages),
                "n_runs": len(group),
            }
        )
    return summaries


def _per_class_table_rows(
    dataset: str, summaries: list[dict[str, Any]]
) -> list[dict[str, str | int]]:
    by_key = {(row["class_id"], row["method"]): row for row in summaries if row["dataset"] == dataset}
    class_ids = sorted({class_id for class_id, _ in by_key})
    rows: list[dict[str, str | int]] = []
    for class_id in class_ids:
        method_rows = [by_key[(class_id, method)] for method in METHOD_ORDER if (class_id, method) in by_key]
        counts = {row["test_count"] for row in method_rows}
        if len(counts) != 1:
            raise ValueError(f"{dataset}/class_{class_id}: methods disagree on test count.")
        row: dict[str, str | int] = {"class": class_id, "test_count": counts.pop()}
        for method in METHOD_ORDER:
            value = by_key.get((class_id, method))
            row[METHOD_LABELS[method]] = (
                _format_mean_std(value["coverage_mean"], value["coverage_std"], percent=True)
                if value is not None
                else "--"
            )
        rows.append(row)
    return rows


def _write_main_results_tex(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a publication-formatted primary comparison table."""
    fields = (
        "Dataset",
        "Method",
        "Marginal coverage (\\%)",
        "Macro class coverage (\\%)",
        "Worst-class coverage (\\%)",
        "Mean set size",
        "Full-set rate (\\%)",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("% Generated by aggregate_conference_results.py; five end-to-end seeds.\n")
        stream.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        stream.write(" & ".join(fields) + " \\\\\n\\midrule\n")
        for row in rows:
            values = (
                row["dataset"], row["method"], row["marginal_coverage"],
                row["macro_class_coverage"], row["worst_class_coverage"],
                row["mean_set_size"], row["full_set_rate"],
            )
            stream.write(" & ".join(values) + " \\\\\n")
        stream.write("\\bottomrule\n\\end{tabular}\n")


def _write_ablation_results_tex(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Write the focused OCQR ablation table from five-seed summaries."""
    rows = sorted(
        summaries,
        key=lambda row: (
            _dataset_sort_key(row["dataset"]),
            ABLATION_ORDER.index(row["method"])
            if row["method"] in ABLATION_ORDER
            else len(ABLATION_ORDER),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("% Generated by aggregate_conference_results.py; five end-to-end seeds.\n")
        stream.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
        stream.write(
            "Method & Marginal (\\%) & Macro (\\%) & Worst (\\%) & Size & "
            "CCR (\\%) & SFS & MDJ & Empty (\\%) \\\\\n\\midrule\n"
        )
        for row in rows:
            values = (
                ABLATION_LABELS.get(row["method"], row["method"]),
                _format_mean_std(row["marginal_coverage_mean"], row["marginal_coverage_std"], percent=True),
                _format_mean_std(row["macro_class_coverage_mean"], row["macro_class_coverage_std"], percent=True),
                _format_mean_std(row["worst_class_coverage_mean"], row["worst_class_coverage_std"], percent=True),
                _format_mean_std(row["mean_set_size_mean"], row["mean_set_size_std"], percent=False),
                _format_mean_std(row["ccr_mean"], row["ccr_std"], percent=True),
                _format_mean_std(row["avg_sfs_mean"], row["avg_sfs_std"], percent=False),
                _format_mean_std(row["avg_mdj_mean"], row["avg_mdj_std"], percent=False),
                _format_mean_std(row["raw_empty_rate_mean"], row["raw_empty_rate_std"], percent=True),
            )
            stream.write(" & ".join(values) + " \\\\\n")
        stream.write("\\bottomrule\n\\end{tabular}\n")


def _write_per_class_tex(path: Path, dataset: str, rows: list[dict[str, str | int]]) -> None:
    """Write a publication-formatted per-class coverage table for one dataset."""
    fields = ("Class", "Test count", "LAC (\\%)", "APS (\\%)", "OAPS (\\%)", "COPOC (\\%)", "OCQR (\\%)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "% Generated by aggregate_conference_results.py; coverage is mean $\\pm$ standard deviation over five seeds.\n"
        )
        stream.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n")
        stream.write(" & ".join(fields) + " \\\\\n\\midrule\n")
        for row in rows:
            values = (
                str(row["class"]), str(row["test_count"]), str(row["LAC"]),
                str(row["APS"]), str(row["OAPS"]), str(row["COPOC"]), str(row["OCQR"]),
            )
            stream.write(" & ".join(values) + " \\\\\n")
        stream.write("\\bottomrule\n\\end{tabular}\n")


def _summarize(
    rows: list[dict[str, Any]], metrics: tuple[str, ...] = PRIMARY_METRICS
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    summaries = []
    for (dataset, method), group in sorted(grouped.items()):
        if {row["seed"] for row in group} != SEEDS or len(group) != len(SEEDS):
            raise ValueError(f"{dataset}/{method}: exactly one complete five-seed run set is required.")
        summary = {"dataset": dataset, "method": method, "target_coverage": 0.90}
        summary["n_runs"] = len(group)
        for metric in metrics:
            values = [row[metric] for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values)
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = sorted(path.parent for path in args.runs.rglob("provenance.json"))
    if not run_dirs:
        raise SystemExit("No canonical run directories found; refusing to create empty tables.")

    run_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    split_hashes: dict[str, set[str]] = defaultdict(set)
    protocol_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for run in run_dirs:
        provenance, metrics = _validate_run(run)
        row = {"dataset": provenance["dataset"], "method": provenance["method"],
               "seed": provenance["seed"], **metrics["aggregate"]}
        (ablation_rows if provenance["method"].startswith("ocqr_") else run_rows).append(row)
        split_hashes[provenance["dataset"]].add(provenance["split_hash"])
        protocol_hashes[(provenance["dataset"], provenance["method"])].add(provenance["protocol_hash"])
        for entry in metrics["per_class"]:
            class_rows.append({"dataset": provenance["dataset"], "method": provenance["method"],
                               "seed": provenance["seed"], **entry})
        for entry in _read_json(run / "calibration.json").get("classes", []):
            diagnostic_rows.append({"dataset": provenance["dataset"], "method": provenance["method"],
                                    "seed": provenance["seed"], **entry})
    for dataset, hashes in split_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"{dataset}: methods/seeds have mismatched frozen split hashes.")
    for key, hashes in protocol_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"{key[0]}/{key[1]}: seeds have mismatched frozen protocol hashes.")

    summaries = _summarize(run_rows)
    ablation_datasets = {row["dataset"] for row in ablation_rows}
    canonical_ablation_rows = [
        row
        for row in run_rows
        if row["dataset"] in ablation_datasets and row["method"] == "ocqr"
    ]
    ablation_artifact_rows = ablation_rows + canonical_ablation_rows
    ablation_summaries = (
        _summarize(ablation_artifact_rows, ABLATION_METRICS)
        if ablation_rows
        else []
    )
    args.results.mkdir(parents=True, exist_ok=True)
    _write_csv(args.results / "main_results.csv", summaries, sorted({key for row in summaries for key in row}))
    _write_json(args.results / "main_results.json", summaries)
    _write_csv(args.results / "run_results.csv", run_rows, sorted({key for row in run_rows for key in row}))
    _write_csv(args.results / "per_class_results.csv", class_rows, sorted({key for row in class_rows for key in row}))
    _write_csv(args.results / "ablation_results.csv", ablation_summaries,
               sorted({key for row in ablation_summaries for key in row}) or ["dataset", "method", "target_coverage"])
    _write_csv(args.results / "ablation_run_results.csv", ablation_artifact_rows,
               sorted({key for row in ablation_artifact_rows for key in row}) or ["dataset", "method", "seed"])
    _write_csv(args.results / "calibration_diagnostics.csv", diagnostic_rows, sorted({key for row in diagnostic_rows for key in row}))
    per_class_summaries = _summarize_per_class(class_rows)
    _write_csv(
        args.results / "per_class_summary.csv",
        per_class_summaries,
        ["dataset", "method", "class_id", "test_count", "coverage_mean", "coverage_std", "n_runs"],
    )
    _write_json(args.results / "per_class_summary.json", per_class_summaries)
    main_table_rows = _main_table_rows(summaries)
    _write_main_results_tex(args.results / "tables" / "main_results.tex", main_table_rows)
    for dataset in sorted({row["dataset"] for row in ablation_summaries}, key=_dataset_sort_key):
        _write_ablation_results_tex(
            args.results / "tables" / f"{dataset}_ocqr_ablation.tex",
            [row for row in ablation_summaries if row["dataset"] == dataset],
        )
    for dataset in sorted({row["dataset"] for row in summaries}, key=_dataset_sort_key):
        dataset_main_rows = [row for row in main_table_rows if row["dataset"] == DATASET_LABELS.get(dataset, dataset)]
        _write_main_results_tex(
            args.results / "tables" / f"{dataset}_main_results.tex", dataset_main_rows
        )
        _write_per_class_tex(
            args.results / "tables" / f"{dataset}_per_class_coverage.tex",
            dataset,
            _per_class_table_rows(dataset, per_class_summaries),
        )
    _write_json(
        args.results / "aggregation.json",
        {
            "ablation_summary_count": len(ablation_summaries),
            "main_summary_count": len(summaries),
            "method_version": METHOD_VERSION,
            "per_class_summary_count": len(per_class_summaries),
            "run_count": len(run_dirs),
            "schema_version": "conference-v0.3-aggregation-v3",
        },
    )


if __name__ == "__main__":
    main()
