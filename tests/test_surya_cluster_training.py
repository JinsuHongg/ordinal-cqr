"""Focused tests for reproducible Surya cluster training infrastructure."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


ROOT = Path(__file__).parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load_script(
    "surya_preflight", ROOT / "scripts/experiments/check_surya_setup.py"
)
TRAINING = _load_script("surya_training", ROOT / "scripts/experiments/training.py")
STATS = _load_script(
    "surya_channel_stats", ROOT / "scripts/experiments/compute_surya_channel_stats.py"
)


def _write_split(path: Path, timestamp: str, intensity: float, label: str) -> str:
    pd.DataFrame(
        {
            "timestamp": [timestamp],
            "max_intensity": [intensity],
            "max_goes_class": [label],
        }
    ).to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_config(index_dir: Path) -> OmegaConf:
    split_values = {
        "train": ("2018-01-01", 1e-6, "C1.0"),
        "val": ("2018-03-01", 1e-7, "B1.0"),
        "cal": ("2018-05-01", 1e-5, "M1.0"),
        "test": ("2020-01-01", 1e-4, "X1.0"),
    }
    hashes = {}
    for split, values in split_values.items():
        hashes[split] = _write_split(index_dir / f"{split}.csv", *values)
    return OmegaConf.create(
        {
            "data": {
                "flare_index": {
                    **{split: f"{split}.csv" for split in split_values},
                    "sha256": hashes,
                },
                "ordinal_label_type": "max_goes_class",
                "filter_numeric_target_column": "max_intensity",
                "fq_max_intensity_exclusion_threshold": 1e-7,
                "excluded_goes_classes": ["M0.9"],
                "batch_size": 2,
            },
            "trainer": {"max_epochs": 3},
        }
    )


def test_stats_groups_timestamp_series_by_calendar_year() -> None:
    groups = STATS._group_timestamps_by_year(
        pd.Series(["2010-01-01", "2010-02-01", "2011-01-01"])
    )

    assert set(groups) == {"2010", "2011"}
    assert len(groups["2010"]) == 2
    assert len(groups["2011"]) == 1


def test_stats_selects_unambiguous_requested_positions() -> None:
    available = pd.to_datetime(
        ["2010-01-01", "2010-01-01", "2010-02-01", "2010-03-01"]
    )
    requested = pd.to_datetime(["2010-01-01", "2010-02-01"])

    positions = STATS._unambiguous_requested_positions(available, requested)

    assert positions.tolist() == [2]


def test_split_audit_validates_hashes_targets_and_chronology(tmp_path: Path) -> None:
    cfg = _audit_config(tmp_path)
    audit, errors = PREFLIGHT.build_split_audit(cfg, tmp_path)

    assert errors == []
    assert audit["validation"]["passed"] is True
    assert audit["future_test_strictly_after_development"] is True
    assert audit["planned_optimizer_steps_upper_bound_before_image_filtering"] == 3
    assert all(
        item["direct_timestamp_overlap_count"] == 0
        for item in audit["pairwise_temporal_checks"].values()
    )


def test_split_audit_rejects_direct_timestamp_overlap(tmp_path: Path) -> None:
    cfg = _audit_config(tmp_path)
    validation = pd.read_csv(tmp_path / "val.csv")
    validation["timestamp"] = "2018-01-01"
    validation.to_csv(tmp_path / "val.csv", index=False)
    cfg.data.flare_index.sha256.val = hashlib.sha256(
        (tmp_path / "val.csv").read_bytes()
    ).hexdigest()

    audit, errors = PREFLIGHT.build_split_audit(cfg, tmp_path)

    assert audit["validation"]["passed"] is False
    assert any("train/val have 1 directly overlapping timestamps" in error for error in errors)


def test_training_wrapper_applies_configured_seed(monkeypatch) -> None:
    calls = []
    fake_trainer = object()
    cfg = OmegaConf.create({"seed": 4})
    monkeypatch.delenv("SURYA_RUN_DIR", raising=False)
    monkeypatch.setattr(TRAINING.L, "seed_everything", lambda seed, workers: calls.append((seed, workers)))
    monkeypatch.setattr(TRAINING, "_fit", lambda received: fake_trainer)

    TRAINING.train.__wrapped__(cfg)

    assert calls == [(4, True)]


def test_protocol_hash_excludes_checkpoint_output_path() -> None:
    first = OmegaConf.create(
        {"seed": 0, "model": {"save_ckpt_path": "/run/a", "type": "resnet18"}}
    )
    second = OmegaConf.create(
        {"seed": 0, "model": {"save_ckpt_path": "/run/b", "type": "resnet18"}}
    )

    _, first_exact, first_protocol = TRAINING._resolved_config_payload(first)
    _, second_exact, second_protocol = TRAINING._resolved_config_payload(second)

    assert first_exact != second_exact
    assert first_protocol == second_protocol


def test_surya_training_configs_share_retained_population_and_determinism() -> None:
    configs = [
        ROOT / "configs/qr/QR_resnet18_train_surya_bench.yaml",
        ROOT / "configs/cls/CLS_resnet18_train_surya_bench.yaml",
        ROOT / "configs/cls/CLS_resnet18_binomial_train_surya_bench.yaml",
        ROOT / "configs/cls/CLS_resnet18_copoc_train_surya_bench.yaml",
    ]
    for path in configs:
        cfg = OmegaConf.load(path)
        assert cfg.seed == 0
        assert cfg.data.fq_max_intensity_exclusion_threshold == 1e-7
        assert list(cfg.data.excluded_goes_classes) == ["M0.9"]
        assert cfg.trainer.deterministic is True
        assert set(cfg.data.flare_index.sha256) == {"train", "val", "cal", "test"}

    qr = OmegaConf.load(configs[0])
    assert qr.model.loss.type == "pinball"


def test_slurm_driver_routes_seed_to_unique_job_directory() -> None:
    script = (ROOT / "scripts/slurm/surya_train.sbatch").read_text()
    assert 'SEED="${SURYA_SEED:-0}"' in script
    assert 'training/job_$JOB_TOKEN' in script
    assert '"seed=$SEED"' in script
    assert '"model.save_ckpt_path=$RUN_DIR/checkpoints"' in script
    assert '"trainer.deterministic=true"' in script


def test_surya_calibration_uses_explicit_checkpoint_path() -> None:
    script = (ROOT / "scripts/slurm/surya_calibrate.sbatch").read_text()

    assert '[[ ! -f "$SURYA_CHECKPOINT" ]]' in script
    assert 'CHECKPOINT_PATH="$(realpath "$SURYA_CHECKPOINT")"' in script
    assert 'BASE_OVERRIDE="check_point.base=$CHECKPOINT_DIR"' in script
    assert 'OVERRIDE="check_point.qr=$CHECKPOINT_NAME"' in script
