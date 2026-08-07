"""Contract tests for strict conference-result aggregation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/experiments/aggregate_conference_results.py"
SPEC = importlib.util.spec_from_file_location("conference_aggregation", SCRIPT)
assert SPEC and SPEC.loader
AGGREGATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATION)


def _write_run(root: Path, seed: int, split_hash: str = "split-hash") -> None:
    run = root / "retinamnist" / "ocqr" / f"seed_{seed}"
    run.mkdir(parents=True)
    config = f"seed: {seed}\nmethod: ocqr\n"
    (run / "config.yaml").write_text(config)
    config_hash = hashlib.sha256(config.encode()).hexdigest()
    provenance = {
        "dataset": "retinamnist", "dataset_contract_version": "0.2.0",
        "method": "ocqr", "method_version": "0.3.0", "alpha": 0.1,
        "seed": seed, "split_identifier": "frozen-v1", "split_hash": split_hash,
        "configuration_hash": config_hash, "protocol_hash": "shared-protocol-hash",
        "code_commit": "a" * 40, "checkpoint_identifier": "checkpoint.pt",
        "training_criterion": "pinball loss",
        "checkpoint_selection_criterion": "validation pinball loss",
        "runtime_seconds": 1.0, "hardware": {"accelerator": "cpu"},
        "timestamp": "2026-08-07T00:00:00+00:00",
    }
    metrics = {
        "aggregate": {
            "marginal_coverage": 0.9, "macro_class_coverage": 0.9,
            "worst_class_coverage": 0.9, "mean_set_size": 2.0,
            "full_set_rate": 0.0,
        },
        "per_class": [{"class_id": 0, "count": 1, "coverage": 1.0}],
    }
    prediction = {
        "sample_id": f"sample-{seed}", "Y_ord": 0, "Z": 0.0,
        "prediction_set_raw": [0], "prediction_set_final": [0],
    }
    (run / "provenance.json").write_text(json.dumps(provenance))
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "calibration.json").write_text(json.dumps({"classes": [{"q_k": "+inf"}]}))
    (run / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")


def test_aggregation_writes_five_seed_primary_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    for seed in range(5):
        _write_run(runs, seed)
    results = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["aggregate", "--runs", str(runs), "--results", str(results)])

    AGGREGATION.main()

    row = next(iter(__import__("csv").DictReader((results / "main_results.csv").open())))
    assert row["target_coverage"] == "0.9"
    assert row["worst_class_coverage_mean"] == "0.9"


def test_aggregation_rejects_mismatched_split_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    for seed in range(5):
        _write_run(runs, seed, split_hash="different" if seed == 4 else "shared")
    monkeypatch.setattr(sys, "argv", ["aggregate", "--runs", str(runs), "--results", str(tmp_path / "results")])

    with pytest.raises(ValueError, match="mismatched frozen split hashes"):
        AGGREGATION.main()
