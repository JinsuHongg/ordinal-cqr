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
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_VERSION = "0.3.0"
SEEDS = {0, 1, 2, 3, 4}
PRIMARY_METRICS = (
    "marginal_coverage",
    "macro_class_coverage",
    "worst_class_coverage",
    "mean_set_size",
    "full_set_rate",
)
REQUIRED_PROVENANCE = {
    "dataset", "dataset_contract_version", "method", "method_version", "alpha",
    "seed", "split_identifier", "split_hash", "configuration_hash", "protocol_hash", "code_commit",
    "checkpoint_identifier", "training_criterion", "checkpoint_selection_criterion",
    "runtime_seconds", "hardware", "timestamp",
}
REQUIRED_PREDICTION_FIELDS = {
    "sample_id", "Y_ord", "Z", "prediction_set_raw", "prediction_set_final",
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


def _validate_predictions(path: Path) -> None:
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
        _finite_number(row["Z"], f"{path}:{line_number}: numeric target")
        seen.add(row["sample_id"])
        for key in ("prediction_set_raw", "prediction_set_final"):
            value = row[key]
            if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
                raise ValueError(f"{path}:{line_number}: invalid {key}.")
        _validate_infinity_encoding(row, path)


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
    _validate_predictions(_prediction_path(run))
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


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    summaries = []
    for (dataset, method), group in sorted(grouped.items()):
        if {row["seed"] for row in group} != SEEDS or len(group) != len(SEEDS):
            raise ValueError(f"{dataset}/{method}: exactly one complete five-seed run set is required.")
        summary = {"dataset": dataset, "method": method, "target_coverage": 0.90}
        for metric in PRIMARY_METRICS:
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
    ablation_summaries = _summarize(ablation_rows) if ablation_rows else []
    args.results.mkdir(parents=True, exist_ok=True)
    _write_csv(args.results / "main_results.csv", summaries, sorted({key for row in summaries for key in row}))
    _write_csv(args.results / "run_results.csv", run_rows, sorted({key for row in run_rows for key in row}))
    _write_csv(args.results / "per_class_results.csv", class_rows, sorted({key for row in class_rows for key in row}))
    _write_csv(args.results / "ablation_results.csv", ablation_summaries,
               sorted({key for row in ablation_summaries for key in row}) or ["dataset", "method", "target_coverage"])
    _write_csv(args.results / "ablation_run_results.csv", ablation_rows,
               sorted({key for row in ablation_rows for key in row}) or ["dataset", "method", "seed"])
    _write_csv(args.results / "calibration_diagnostics.csv", diagnostic_rows, sorted({key for row in diagnostic_rows for key in row}))
    table_fields = ["dataset", "method", "target_coverage", *[f"{metric}_mean" for metric in PRIMARY_METRICS], *[f"{metric}_std" for metric in PRIMARY_METRICS]]
    _write_tex(args.results / "tables" / "main_results.tex", summaries, table_fields)
    _write_tex(args.results / "tables" / "per_class_results.tex", class_rows, ["dataset", "method", "seed", "class_id", "count", "coverage"])
    with (args.results / "aggregation.json").open("w", encoding="utf-8") as stream:
        json.dump({"schema_version": "conference-v0.3-aggregation-v2", "method_version": METHOD_VERSION,
                   "run_count": len(run_dirs), "main_summary_count": len(summaries),
                   "ablation_summary_count": len(ablation_summaries)}, stream,
                  allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
