import json
import tempfile
from pathlib import Path

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from ordinal_cqr.explainability.poshoc_uc import OrdinalCQRWrapper
from ordinal_cqr.metrics.classification_metrics import OrdinalCQRMetrics
from scripts.experiments.calibration import (
    build_ordinal_cqr_calibration_payload,
    build_ordinal_cqr_evaluation_payload,
    _split_provenance,
    save_ordinal_cqr_calibration_metadata,
    save_ordinal_cqr_evaluation_metrics,
)


class IdentityQuantileModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _calibrated_wrapper() -> OrdinalCQRWrapper:
    wrapper = OrdinalCQRWrapper(
        IdentityQuantileModel(),
        num_classes=3,
        class_mapping={"low": 0, "middle": 1, "high": 2},
        thresholds=[0.5, 1.5],
        alpha=0.5,
        class_wise=True,
    )
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([0, 0, 1]),
        ),
        batch_size=3,
    )
    wrapper.calibrate(loader)
    return wrapper


def test_metadata_getter_requires_completed_mondrian_calibration() -> None:
    wrapper = OrdinalCQRWrapper(
        IdentityQuantileModel(),
        num_classes=2,
        class_mapping={"low": 0, "high": 1},
        thresholds=[0.5],
        class_wise=True,
    )
    try:
        wrapper.get_calibration_metadata()
    except RuntimeError as error:
        assert "must be calibrated" in str(error)
    else:
        raise AssertionError("Expected metadata access before calibration to raise.")


def test_payload_is_strict_json_and_saves_atomically() -> None:
    wrapper = _calibrated_wrapper()
    cfg = OmegaConf.create(
        {
            "seed": 17,
            "data": {
                "repo": "synthetic",
                "label_type": "class_index",
            },
            "uc": {"csv_path": "unused"},
        }
    )

    payload, config_hash = build_ordinal_cqr_calibration_payload(
        wrapper, cfg, "checkpoint.ckpt"
    )
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

    assert len(config_hash) == 64
    assert payload["classes"][2]["q_hat"] is None
    assert payload["classes"][2]["q_hat_is_infinite"] is True
    assert payload["provenance"]["calibration_split"]["hash_status"] == (
        "stable_manifest_unavailable"
    )
    assert "Infinity" not in encoded
    assert "NaN" not in encoded

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "metadata.json"
        save_ordinal_cqr_calibration_metadata(path, payload)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == payload
        assert not path.with_suffix(".json.tmp").exists()


def test_metadata_buffers_survive_state_dict_round_trip() -> None:
    calibrated = _calibrated_wrapper()
    restored = OrdinalCQRWrapper(
        IdentityQuantileModel(),
        num_classes=3,
        class_mapping={"low": 0, "middle": 1, "high": 2},
        thresholds=[0.5, 1.5],
        alpha=0.5,
        class_wise=True,
    )

    restored.load_state_dict(calibrated.state_dict())
    metadata = restored.get_calibration_metadata()

    assert metadata.counts.tolist() == [2, 1, 0]
    assert metadata.requested_ranks.tolist() == [2, 1, 1]
    assert metadata.empty.tolist() == [False, False, True]


def test_nonzero_rank_does_not_write_metadata() -> None:
    original_rank = rank_zero_only.rank
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.json"
        try:
            rank_zero_only.rank = 1
            save_ordinal_cqr_calibration_metadata(path, {"ok": True})
        finally:
            rank_zero_only.rank = original_rank
        assert not path.exists()


def test_solar_split_provenance_hashes_the_configured_csv() -> None:
    with tempfile.TemporaryDirectory() as directory:
        split = Path(directory) / "calibration.csv"
        split.write_text("timestamp,label\n2020-01-01,C1.0\n", encoding="utf-8")
        cfg = OmegaConf.create(
            {
                "data": {
                    "flare_index": {"path": directory, "cal": split.name}
                }
            }
        )

        provenance = _split_provenance(cfg)

    assert provenance["hash_status"] == "available"
    assert provenance["hash_kind"] == "source_index_file"
    assert provenance["sha256"] is not None
    assert len(provenance["sha256"]) == 64


def test_evaluation_payload_is_strict_json_and_links_calibration() -> None:
    wrapper = _calibrated_wrapper()
    metric = OrdinalCQRMetrics(num_classes=3)
    metric.update(
        torch.tensor([[True, False, True], [False, False, False]]),
        torch.ones((2, 3), dtype=torch.bool),
        torch.tensor([0, 2]),
    )
    wrapper.evaluation_metrics = metric.compute()
    cfg = OmegaConf.create({"data": {"repo": "synthetic"}})

    with tempfile.TemporaryDirectory() as directory:
        calibration_path = Path(directory) / "calibration.json"
        calibration_path.write_text('{"schema_version":"test"}\n', encoding="utf-8")
        payload = build_ordinal_cqr_evaluation_payload(
            wrapper, calibration_path, "a" * 64, cfg
        )
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        evaluation_path = Path(directory) / "evaluation.json"
        save_ordinal_cqr_evaluation_metrics(evaluation_path, payload)
        loaded = json.loads(evaluation_path.read_text(encoding="utf-8"))

    assert loaded == payload
    assert payload["calibration_metadata"]["status"] == "available"
    assert len(payload["calibration_metadata"]["sha256"]) == 64
    assert payload["per_class"][1]["coverage"] is None
    assert payload["per_class"][1]["avg_set_size"] is None
    assert "Infinity" not in encoded
    assert "NaN" not in encoded


def test_evaluation_payload_rejects_empty_test_results() -> None:
    wrapper = _calibrated_wrapper()
    wrapper.evaluation_metrics = OrdinalCQRMetrics(num_classes=3).compute()
    cfg = OmegaConf.create({"data": {"repo": "synthetic"}})

    try:
        build_ordinal_cqr_evaluation_payload(
            wrapper, Path("missing-calibration.json"), "a" * 64, cfg
        )
    except RuntimeError as error:
        assert "empty test set" in str(error)
    else:
        raise AssertionError("Expected empty evaluation metrics to raise.")
