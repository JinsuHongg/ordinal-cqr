"""Validate canonical run artifacts and generate conference CSV/LaTex tables.

The script intentionally rejects incomplete or legacy payloads. It performs no
statistical imputation and never invents experiment results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_VERSION = "0.3.0"
REQUIRED_PROVENANCE = {
    "dataset", "dataset_contract_version", "method", "method_version", "alpha",
    "seed", "split_identifier", "split_hash", "configuration_hash", "code_commit",
    "checkpoint_identifier", "training_criterion", "checkpoint_selection_criterion",
    "runtime_seconds", "hardware", "timestamp",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _validate_run(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (run / "config.yaml").is_file():
        raise ValueError(f"{run}: resolved config.yaml is required")
    provenance, metrics = (_read_json(run / name) for name in (
        "provenance.json", "metrics.json"))
    missing = REQUIRED_PROVENANCE - provenance.keys()
    if missing:
        raise ValueError(f"{run}: missing provenance fields {sorted(missing)}")
    if provenance["method_version"] != METHOD_VERSION:
        raise ValueError(f"{run}: method version is not canonical {METHOD_VERSION}")
    has_predictions = any((run / f"predictions{suffix}").is_file() for suffix in (".parquet", ".csv", ".jsonl"))
    if not (run / "calibration.json").is_file() or not has_predictions:
        raise ValueError(f"{run}: calibration.json and sample-level predictions are required")
    return provenance, metrics


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write a compact generated table; presentation styling belongs in LaTex."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{" + "l" * len(fields) + "}\n\\toprule\n")
        stream.write(" & ".join(field.replace("_", "\\_") for field in fields) + " \\\\\n\\midrule\n")
        for row in rows:
            stream.write(" & ".join(str(row.get(field, "")).replace("_", "\\_") for field in fields) + " \\\\\n")
        stream.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = sorted(path.parent for path in args.runs.rglob("provenance.json"))
    if not run_dirs:
        raise SystemExit("No canonical run directories found; refusing to create empty tables.")

    main_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for run in run_dirs:
        provenance, metrics = _validate_run(run)
        aggregate = metrics.get("aggregate")
        per_class = metrics.get("per_class")
        if not isinstance(aggregate, dict) or not isinstance(per_class, list):
            raise ValueError(f"{run}: metrics must provide aggregate and per_class values.")
        row = {"dataset": provenance["dataset"], "method": provenance["method"],
               "seed": provenance["seed"], **aggregate}
        (ablation_rows if provenance["method"].startswith("ocqr_") else main_rows).append(row)
        for entry in per_class:
            class_rows.append({"dataset": provenance["dataset"], "method": provenance["method"],
                               "seed": provenance["seed"], **entry})
        calibration = _read_json(run / "calibration.json")
        for entry in calibration.get("classes", []):
            diagnostic_rows.append({"dataset": provenance["dataset"], "method": provenance["method"],
                                    "seed": provenance["seed"], **entry})

    args.results.mkdir(parents=True, exist_ok=True)
    _write_csv(args.results / "main_results.csv", main_rows,
               sorted({key for row in main_rows for key in row}))
    _write_csv(args.results / "per_class_results.csv", class_rows,
               sorted({key for row in class_rows for key in row}))
    _write_csv(args.results / "ablation_results.csv", ablation_rows,
               sorted({key for row in ablation_rows for key in row}) or ["dataset", "method", "seed"])
    _write_csv(args.results / "calibration_diagnostics.csv", diagnostic_rows,
               sorted({key for row in diagnostic_rows for key in row}))
    table_fields = ["dataset", "method", "seed", "marginal_coverage", "avg_set_size"]
    _write_tex(args.results / "tables" / "main_results.tex", main_rows, table_fields)
    _write_tex(args.results / "tables" / "per_class_results.tex", class_rows,
               ["dataset", "method", "seed", "class_id", "count", "coverage"])
    _write_tex(args.results / "tables" / "ablation_results.tex", ablation_rows, table_fields)
    with (args.results / "aggregation.json").open("w", encoding="utf-8") as stream:
        json.dump({"schema_version": "conference-v0.3-aggregation-v1", "method_version": METHOD_VERSION,
                   "run_count": len(run_dirs)}, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
