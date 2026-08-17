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
        "point_prediction": 0,
        "prediction_set_raw": [0], "prediction_set_final": [0],
    }
    (run / "provenance.json").write_text(json.dumps(provenance))
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "calibration.json").write_text(json.dumps({"classes": [{"q_k": "+inf"}]}))
    (run / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    (run / "manifest_reference.json").write_text(json.dumps({
        "manifest_sha256": split_hash, "split_counts": {"test": 1},
    }))
    (run / "run_status.json").write_text(json.dumps({
        "status": "evaluation_complete", "stage": "complete",
    }))


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
    assert row["n_runs"] == "5"
    assert (results / "main_results.json").is_file()
    assert (results / "per_class_summary.csv").is_file()
    main_table = (results / "tables" / "retinamnist_main_results.tex").read_text()
    assert "Marginal coverage" in main_table
    assert "$\\pm$" in main_table


def test_aggregation_recovers_missing_ocqr_structural_metrics(tmp_path: Path) -> None:
    """Older canonical runs remain usable when predictions retain the metrics."""
    runs = tmp_path / "runs"
    _write_run(runs, 0)

    _, metrics = AGGREGATION._validate_run(runs / "retinamnist" / "ocqr" / "seed_0")

    aggregate = metrics["aggregate"]
    assert aggregate["ccr"] == 1.0
    assert aggregate["avg_sfs"] == 1.0
    assert aggregate["avg_mdj"] == 0.0
    assert aggregate["raw_empty_rate"] == 0.0


def test_ablation_table_includes_structural_metrics(tmp_path: Path) -> None:
    summaries = [
        {
            "dataset": "retinamnist",
            "method": "ocqr_no_hull",
            "marginal_coverage_mean": 0.9,
            "marginal_coverage_std": 0.01,
            "macro_class_coverage_mean": 0.9,
            "macro_class_coverage_std": 0.01,
            "worst_class_coverage_mean": 0.8,
            "worst_class_coverage_std": 0.02,
            "mean_set_size_mean": 2.0,
            "mean_set_size_std": 0.1,
            "ccr_mean": 0.8,
            "ccr_std": 0.02,
            "avg_sfs_mean": 1.2,
            "avg_sfs_std": 0.1,
            "avg_mdj_mean": 0.2,
            "avg_mdj_std": 0.1,
            "raw_empty_rate_mean": 0.0,
            "raw_empty_rate_std": 0.0,
        }
    ]

    table = tmp_path / "ablation.tex"
    AGGREGATION._write_ablation_results_tex(table, summaries)

    rendered = table.read_text()
    assert "OCQR-NoHull" in rendered
    assert "CCR" in rendered
    assert "SFS" in rendered


def test_aggregation_rejects_mismatched_split_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    for seed in range(5):
        _write_run(runs, seed, split_hash="different" if seed == 4 else "shared")
    monkeypatch.setattr(sys, "argv", ["aggregate", "--runs", str(runs), "--results", str(tmp_path / "results")])

    with pytest.raises(ValueError, match="mismatched frozen split hashes"):
        AGGREGATION.main()


def test_aggregation_rejects_incomplete_run(tmp_path: Path) -> None:
    _write_run(tmp_path, 0)
    run = tmp_path / "retinamnist" / "ocqr" / "seed_0"
    (run / "run_status.json").write_text(json.dumps({"status": "started", "stage": "training"}))

    with pytest.raises(ValueError, match="status is not evaluation_complete"):
        AGGREGATION._validate_run(run)


def test_aggregation_rejects_prediction_count_mismatch(tmp_path: Path) -> None:
    _write_run(tmp_path, 0)
    run = tmp_path / "retinamnist" / "ocqr" / "seed_0"
    manifest = json.loads((run / "manifest_reference.json").read_text())
    manifest["split_counts"]["test"] = 2
    (run / "manifest_reference.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="prediction count"):
        AGGREGATION._validate_run(run)


def test_aggregation_rejects_legacy_copoc_provenance(tmp_path: Path) -> None:
    _write_run(tmp_path, 0)
    run = tmp_path / "retinamnist" / "ocqr" / "seed_0"
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["method"] = "copoc"  # Historical Binomial-LAC artifacts lack the Eq. (5)+APS record.
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match=r"not canonical Eq. \(5\) \+ APS"):
        AGGREGATION._validate_run(run)


def test_aggregation_rejects_legacy_oaps_provenance(tmp_path: Path) -> None:
    _write_run(tmp_path, 0)
    run = tmp_path / "retinamnist" / "ocqr" / "seed_0"
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["method"] = "oaps"
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="not canonical Lu et al. Algorithm 1"):
        AGGREGATION._validate_run(run)


def test_aggregation_rejects_unverifiable_commit(tmp_path: Path) -> None:
    _write_run(tmp_path, 0)
    run = tmp_path / "retinamnist" / "ocqr" / "seed_0"
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["code_commit"] = "unavailable_backfilled"
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="exact 40-character Git SHA"):
        AGGREGATION._validate_run(run)


def test_aggregation_rejects_explicitly_dirty_run(tmp_path: Path) -> None:
    _write_run(tmp_path, 0)
    run = tmp_path / "retinamnist" / "ocqr" / "seed_0"
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["git_dirty"] = True
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="rejects a dirty source tree"):
        AGGREGATION._validate_run(run)


@pytest.mark.parametrize(
    ("method", "message"),
    [("lac", "not canonical exact split LAC"), ("aps", "not the canonical boundary rule")],
)
def test_aggregation_rejects_legacy_lac_and_aps_provenance(
    tmp_path: Path, method: str, message: str
) -> None:
    _write_run(tmp_path, 0)
    run = tmp_path / "retinamnist" / "ocqr" / "seed_0"
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["method"] = method
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=message):
        AGGREGATION._validate_run(run)


def test_aggregation_accepts_canonical_oaps_provenance(tmp_path: Path) -> None:
    _write_run(tmp_path, 0)
    run = tmp_path / "retinamnist" / "ocqr" / "seed_0"
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["method"] = "oaps"
    provenance["oaps"] = {
        "oaps_method_version": "1.0.0-lu2022-algorithm1",
        "set_family": "greedy_mode_centered_adjacent_expansion",
        "calibration": "pooled_exact_augmented_rank",
        "mode_tie_rule": "lowest_class",
        "adjacent_tie_rule": "upper_right",
    }
    provenance_path.write_text(json.dumps(provenance))

    AGGREGATION._validate_run(run)
