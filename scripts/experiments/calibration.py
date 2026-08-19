import os
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import hydra
from hydra.utils import to_absolute_path
from loguru import logger as lgr_logger
import torch
from omegaconf import OmegaConf
# PyTorch 2.6 security monkey-patch for Hydra dict configs
_original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from ordinal_cqr.datamodules import (
    FlareHelioviewerRegDataModule,
    FlareSuryaBenchDataModule,
)
from ordinal_cqr.explainability import (
    LaplaceWrapper,
    SafeLaplaceModel,
    OrdinalCQRWrapper,
    CPWrapper,
    CQRWrapper,
)
from ordinal_cqr.models import ResNetMCD, ResNetQR
from ordinal_cqr.datasets.flare_cls_datasets import build_ocqr_flare_manifest_audit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _split_provenance(
    cfg, split_name: str = "cal"
) -> dict[str, str | None]:
    flare_index = cfg.data.get("flare_index")
    if flare_index is not None and flare_index.get(split_name) is not None:
        configured = Path(str(flare_index.path)) / str(flare_index.get(split_name))
        resolved = Path(to_absolute_path(str(configured)))
        if resolved.is_file():
            return {
                "identifier": str(configured),
                "sha256": _sha256_file(resolved),
                "hash_status": "available",
                "hash_kind": "source_index_file",
            }
        return {
            "identifier": str(configured),
            "sha256": None,
            "hash_status": "file_not_found",
            "hash_kind": None,
        }
    return {
        "identifier": f"{cfg.data.get('repo', 'unknown')}:{split_name}",
        "sha256": None,
        "hash_status": "stable_manifest_unavailable",
        "hash_kind": None,
    }


def _configuration_sha256(cfg) -> str:
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    canonical_config = json.dumps(
        resolved_config, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()


def build_solar_manifest_audit(cfg) -> dict[str, object]:
    """Build the declared source-to-retained audit for every solar CSV split."""
    flare_index = cfg.data.get("flare_index")
    if flare_index is None:
        raise ValueError("Solar manifest audit requires data.flare_index.")
    split_files = {
        "train": flare_index.get("train"),
        "validation": flare_index.get("val"),
        "calibration": flare_index.get("cal"),
        "test": flare_index.get("test"),
    }
    if any(filename is None for filename in split_files.values()):
        raise ValueError("Solar manifest audit requires train/val/cal/test split files.")
    source_root = Path(to_absolute_path(str(flare_index.path)))
    policy = {
        "ordinal_label_column": str(
            cfg.data.get("ordinal_label_type", "max_goes_class")
        ),
        "numeric_target_column": str(
            cfg.data.get("filter_numeric_target_column", "max_intensity")
        ),
        "fq_max_intensity_exclusion_threshold": cfg.data.get(
            "fq_max_intensity_exclusion_threshold"
        ),
        "excluded_goes_classes": list(cfg.data.get("excluded_goes_classes", [])),
        "filter_stage": "before_image_availability_filtering",
    }
    return {
        "schema_version": "ocqr-solar-manifest-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "splits": {
            split_name: build_ocqr_flare_manifest_audit(
                str(source_root / str(filename)),
                split_name=split_name,
                ordinal_label_column=policy["ordinal_label_column"],
                numeric_target_column=policy["numeric_target_column"],
                fq_max_intensity=policy["fq_max_intensity_exclusion_threshold"],
                excluded_goes_classes=tuple(policy["excluded_goes_classes"]),
            )
            for split_name, filename in split_files.items()
        },
    }


@rank_zero_only
def save_ordinal_cqr_calibration_metadata(
    file_path: Path, payload: dict[str, object]
) -> None:
    """Atomically persist strict JSON metadata from global rank zero."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, file_path)


@rank_zero_only
def save_ordinal_cqr_evaluation_metrics(
    file_path: Path, payload: dict[str, object]
) -> None:
    """Atomically persist strict JSON evaluation metrics from global rank zero."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, file_path)


@rank_zero_only
def save_solar_manifest_audit(file_path: Path, payload: dict[str, object]) -> None:
    """Atomically persist strict JSON source-to-retained solar provenance."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, file_path)


def _json_float(value: torch.Tensor) -> float | None:
    number = float(value.detach().cpu())
    return number if math.isfinite(number) else None


def build_ordinal_cqr_evaluation_payload(
    wrapper: OrdinalCQRWrapper,
    calibration_metadata_path: Path,
    configuration_hash: str,
    cfg,
) -> dict[str, object]:
    """Build strict-JSON evaluation metrics linked to calibration metadata."""
    metrics = wrapper.get_evaluation_metrics()
    num_samples = int(float(metrics["num_samples"].detach().cpu()))
    if num_samples == 0:
        raise RuntimeError("Cannot persist OCQR metrics for an empty test set.")
    vector_keys = {
        "per_class_count",
        "per_class_coverage",
        "per_class_avg_set_size",
    }
    aggregate = {
        name: _json_float(value)
        for name, value in metrics.items()
        if name not in vector_keys and name != "num_samples"
    }
    counts = metrics["per_class_count"].detach().cpu().tolist()
    coverage = metrics["per_class_coverage"].detach().cpu()
    set_sizes = metrics["per_class_avg_set_size"].detach().cpu()
    per_class = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "count": int(counts[class_id]),
            "coverage": _json_float(coverage[class_id]),
            "avg_set_size": _json_float(set_sizes[class_id]),
        }
        for class_id, class_name in enumerate(wrapper.class_names)
    ]
    calibration_hash = (
        _sha256_file(calibration_metadata_path)
        if calibration_metadata_path.is_file()
        else None
    )
    return {
        "schema_version": "ocqr-evaluation-metrics-v1",
        "method": "ocqr",
        "method_version": "0.3.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "alpha": wrapper.alpha,
        "num_classes": wrapper.num_classes,
        "class_mapping": wrapper.class_mapping,
        "thresholds": wrapper.thresholds,
        "interval_convention": "left_closed_right_open",
        "num_samples": num_samples,
        "aggregate": aggregate,
        "per_class": per_class,
        "metric_definitions": {
            "raw_empty_rate": "Pr(|C_raw| = 0)",
            "raw_fragmented_rate": "Pr(SFS(C_raw) > 1)",
            "avg_fallback_inflation": "mean(|C_fallback| - |C_raw|), labels per sample",
            "avg_hull_inflation": "mean(|C_final| - |C_fallback|), labels per sample",
            "avg_total_inflation": "mean(|C_final| - |C_raw|), labels per sample",
            "normalized_inflation": "corresponding mean added labels divided by K",
            "full_set_rate": "Pr(|C_final| = K)",
        },
        "calibration_metadata": {
            "filename": calibration_metadata_path.name,
            "sha256": calibration_hash,
            "status": "available" if calibration_hash is not None else "unavailable",
        },
        "configuration_sha256": configuration_hash,
        "test_split": _split_provenance(cfg, "test"),
    }


def build_ordinal_cqr_calibration_payload(
    wrapper: OrdinalCQRWrapper,
    cfg,
    checkpoint_identifier: str,
    solar_manifest_audit_path: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Combine method-owned calibration state with run-level provenance."""
    config_hash = _configuration_sha256(cfg)
    code_commit, code_dirty = _git_state()
    metadata = wrapper.get_calibration_metadata()
    payload: dict[str, object] = {
        "schema_version": "ocqr-calibration-metadata-v1",
        "method": "ocqr",
        "method_version": "0.3.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "alpha": wrapper.alpha,
        "calibration_mode": "true_label_mondrian",
        "rank_rule": "ceil_n_plus_1",
        "num_classes": wrapper.num_classes,
        "class_mapping": wrapper.class_mapping,
        "thresholds": wrapper.thresholds,
        "interval_convention": "left_closed_right_open",
        "lower_idx": wrapper.lower_idx,
        "upper_idx": wrapper.upper_idx,
        "postprocessing": {
            "apply_empty_set_fallback": wrapper.apply_empty_set_fallback,
            "enforce_ordinal_hull": wrapper.enforce_ordinal_hull,
        },
        "target_bin_contract": {
            "version": cfg.data.get("target_bin_contract_version"),
            "status": (
                "available"
                if cfg.data.get("target_bin_contract_version") is not None
                else "unavailable"
            ),
        },
        "classes": metadata.to_class_records(wrapper.class_names),
        "provenance": {
            "seed": int(cfg.get("seed", 42)),
            "checkpoint_identifier": checkpoint_identifier,
            "configuration_sha256": config_hash,
            "code_commit": code_commit,
            "code_dirty": code_dirty,
            "dataset": str(cfg.data.get("repo", "unknown")),
            "numeric_target_field": str(cfg.data.get("label_type", "unknown")),
            "ordinal_label_field": str(
                cfg.data.get("ordinal_label_type", "class_index")
            ),
            "target_transform": str(cfg.data.get("target_norm_type", "identity")),
            "calibration_split": _split_provenance(cfg),
            "solar_manifest_audit": (
                {
                    "filename": solar_manifest_audit_path.name,
                    "sha256": _sha256_file(solar_manifest_audit_path),
                    "status": "available",
                }
                if solar_manifest_audit_path is not None
                and solar_manifest_audit_path.is_file()
                else {"status": "not_applicable"}
            ),
        },
    }
    return payload, config_hash


def save_batch_to_csv(file_path, batch_dict, header_written=False):
    """
    Helper to save a batch of dictionary results to CSV.
    Handles both vectors (per-sample) and scalars (constants).
    """
    keys = list(batch_dict.keys())

    # Determine Batch Size from the first VECTOR found
    batch_size = 1
    for k in keys:
        val = batch_dict[k]
        if hasattr(val, "ndim") and val.ndim > 0:
            batch_size = len(val)
            break
        elif isinstance(val, list):
            batch_size = len(val)
            break

    rows = []
    for idx in range(batch_size):
        row = {}
        for k in keys:
            val = batch_dict[k]

            if hasattr(val, "ndim") and val.ndim == 0:
                item = val
            elif not hasattr(val, "__getitem__") or isinstance(val, (int, float)):
                item = val
            else:
                if len(val) > idx:
                    item = val[idx]
                else:
                    item = None

            if isinstance(item, torch.Tensor):
                if item.ndim == 0:
                    item = item.item()
                else:
                    item = item.tolist()

            row[k] = item
        rows.append(row)

    # Write to CSV
    mode = "a" if header_written else "w"
    with open(file_path, mode=mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not header_written:
            writer.writeheader()
        writer.writerows(rows)


@hydra.main(
    config_path="../../configs/",
    config_name="resnet34_calibration.yaml",
    version_base=None,
)
def run_uc_cal(cfg):
    L.seed_everything(cfg.get("seed", 42), workers=True)
    methods = cfg.uc.get("methods", ["mcd", "cp", "cqr", "lp"])

    if cfg.data.get("repo") == "retinamnist":
        from ordinal_cqr.datamodules.retina_mnist import RetinaMNISTDataModule
        datamodule = RetinaMNISTDataModule(data_dir="/mnt/storage/medmnist", batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    elif cfg.data.get("repo") == "utkface":
        from ordinal_cqr.datamodules.utkface import UTKFaceDataModule
        datamodule = UTKFaceDataModule(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, thresholds=cfg.uc.get("thresholds", [20.0, 40.0, 60.0, 80.0]), label_type=cfg.data.get("label_type", "ordinal"))
    elif cfg.data.get("repo") == "eyepacs":
        from ordinal_cqr.datamodules.eyepacs import EyePACSDataModule
        datamodule = EyePACSDataModule(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, label_type=cfg.data.get("label_type", "ordinal"))
    elif cfg.data.get("repo") == "adience":
        from ordinal_cqr.datamodules.adience import AdienceDataModule
        datamodule = AdienceDataModule(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, label_type=cfg.data.get("label_type", "ordinal"))
    elif "input_zarr_path" in cfg.data:
        datamodule = FlareSuryaBenchDataModule(cfg=cfg)
    else:
        datamodule = FlareHelioviewerRegDataModule(cfg=cfg)
    datamodule.setup(stage="calibrate")
    datamodule.setup(stage="test")

    if hasattr(datamodule, "cal_dataloader"):
        calibration_loader = datamodule.cal_dataloader()
    else:
        lgr_logger.warning("No cal_dataloader found, using val_dataloader.")
        calibration_loader = datamodule.val_dataloader()

    test_loader = datamodule.test_dataloader()

    # Load Models
    base_path = cfg.check_point.base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcd = None
    qr = None
    qr_pretrained_path = None
    ordinal_metadata_path = None
    ordinal_config_hash = None
    solar_manifest_audit_path = None

    if "ordinal_cqr" in methods and "input_zarr_path" in cfg.data:
        audit_config_hash = _configuration_sha256(cfg)
        solar_manifest_audit_path = Path(to_absolute_path(str(cfg.uc.csv_path))) / (
            f"solar_manifest_audit_{audit_config_hash[:12]}.json"
        )
        save_solar_manifest_audit(
            solar_manifest_audit_path, build_solar_manifest_audit(cfg)
        )
        lgr_logger.info(f"Solar manifest audit saved to {solar_manifest_audit_path}.")

    if any(m in methods for m in ["mcd", "cp", "lp"]):
        if cfg.check_point.mcd is None:
            lgr_logger.warning(
                "MCD checkpoint is null, skipping MCD-dependent methods."
            )
        else:
            mcd_pretrained_path = os.path.join(base_path, "mcd", cfg.check_point.mcd)
            match cfg.check_point.model_type:
                case "resnet":
                    mcd = ResNetMCD.load_from_checkpoint(
                        mcd_pretrained_path, strict=False, weights_only=False
                    )
                    mcd.to(device)
                case _:
                    raise ValueError(f"Wrong model type: {cfg.check_point.model_type}")

    if "cqr" in methods or "ordinal_cqr" in methods:
        qr_ckpt = cfg.check_point.get("qr", cfg.check_point.get("resnet18_qr", None))
        if qr_ckpt is None:
            lgr_logger.warning("QR checkpoint is null, skipping CQR.")
        else:
            qr_pretrained_path = os.path.join(base_path, qr_ckpt)
            match cfg.check_point.model_type:
                case "resnet":
                    qr = ResNetQR.load_from_checkpoint(
                        qr_pretrained_path, strict=False, weights_only=False
                    )
                    qr.to(device)
                case _:
                    raise ValueError(f"Wrong model type: {cfg.check_point.model_type}")

    # Initialize Wrappers
    alpha = cfg.uc.significance_level

    wrappers = {}

    if "cp" in methods and mcd is not None:
        wrappers["cp"] = CPWrapper(
            trained_model=mcd, score_type=cfg.uc.cp.score_type, alpha=alpha
        ).to(device)

    if "cqr" in methods and qr is not None:
        wrappers["cqr"] = CQRWrapper(
            trained_model=qr,
            alpha=alpha,
            lower_idx=cfg.uc.cqr.lower_idx,
            upper_idx=cfg.uc.cqr.upper_idx,
        ).to(device)

    if "ordinal_cqr" in methods and qr is not None:
        if not cfg.uc.get("class_wise", False):
            raise ValueError(
                "Canonical OrdinalCQR metadata requires uc.class_wise=true."
            )
        wrappers["ordinal_cqr"] = OrdinalCQRWrapper(
            qr,
            num_classes=cfg.uc.num_classes,  # Assuming num_classes is in cfg.data
            class_mapping=cfg.uc.class_mapping,
            thresholds=cfg.uc.thresholds,  # Assuming thresholds are defined in config
            alpha=alpha,
            lower_idx=cfg.uc.cqr.lower_idx,
            upper_idx=cfg.uc.cqr.upper_idx,
            class_wise=cfg.uc.get("class_wise", False),
            apply_empty_set_fallback=cfg.uc.get("ordinal_cqr", {}).get(
                "apply_empty_set_fallback", True
            ),
            enforce_ordinal_hull=cfg.uc.get("ordinal_cqr", {}).get(
                "enforce_ordinal_hull", True
            ),
        ).to(device)

    if "lp" in methods and mcd is not None:
        wrappers["lp"] = LaplaceWrapper(
            trained_model=mcd,
            alpha=alpha,
            subset_size=cfg.uc.lp.subset_size,
        ).to(device)

    # Calibration -------------------------------------------------------------
    lgr_logger.info(f"Running Calibration for methods: {methods}")

    if "cp" in wrappers:
        wrappers["cp"].calibrate(calibration_loader)
        lgr_logger.info(f"CP Q_hat: {wrappers['cp'].q_hat.item():.4f}")

    if "cqr" in wrappers:
        wrappers["cqr"].calibrate(calibration_loader)
        lgr_logger.info(f"CQR Q_hat: {wrappers['cqr'].q_hat.item():.4f}")

    if "ordinal_cqr" in wrappers:
        wrappers["ordinal_cqr"].calibrate(calibration_loader)
        checkpoint_identifier = os.path.basename(str(qr_pretrained_path))
        payload, config_hash = build_ordinal_cqr_calibration_payload(
            wrappers["ordinal_cqr"],
            cfg,
            checkpoint_identifier,
            solar_manifest_audit_path=solar_manifest_audit_path,
        )
        metadata_path = Path(to_absolute_path(str(cfg.uc.csv_path))) / (
            f"ordinal_cqr_alpha{alpha}_{config_hash[:12]}_calibration_metadata.json"
        )
        save_ordinal_cqr_calibration_metadata(metadata_path, payload)
        ordinal_metadata_path = metadata_path
        ordinal_config_hash = config_hash
        lgr_logger.info(f"OrdinalCQR calibrated; metadata saved to {metadata_path}.")

    if "lp" in wrappers:
        wrappers["lp"].fit_laplace(calibration_loader)

    # Prediction --------------------------------------------------------------
    lgr_logger.info("Running Prediction on Test Set...")

    # Initialize Loggers
    loggers = []

    # Wandb Logger (using .get with defaults if wandb config is missing)
    if cfg.get("wandb"):
        wandb_logger = WandbLogger(
            project=cfg.wandb.get("project", "default_project"),
            entity=cfg.wandb.get("entity", "default_entity"),
            name=f"calibration_run_{cfg.experiment.task}",
            save_dir=cfg.wandb.get("save_dir", "./wandb_logs"),
        )
        loggers.append(wandb_logger)

    # CSV Logger
    csv_logger = CSVLogger(save_dir=cfg.uc.csv_path, name="summary")
    loggers.append(csv_logger)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=loggers,  # Pass the list of loggers
    )

    results = {}

    if "mcd" in methods and mcd is not None:
        results["mcd"] = trainer.predict(mcd, test_loader)

    if "cp" in wrappers:
        results["cp"] = trainer.predict(wrappers["cp"], test_loader)

    if "cqr" in wrappers:
        results["cqr"] = trainer.predict(wrappers["cqr"], test_loader)

    if "ordinal_cqr" in wrappers:
        trainer.test(wrappers["ordinal_cqr"], test_loader)
        if ordinal_metadata_path is None or ordinal_config_hash is None:
            raise RuntimeError("OrdinalCQR calibration metadata linkage is missing.")
        evaluation_payload = build_ordinal_cqr_evaluation_payload(
            wrappers["ordinal_cqr"],
            ordinal_metadata_path,
            ordinal_config_hash,
            cfg,
        )
        evaluation_path = Path(to_absolute_path(str(cfg.uc.csv_path))) / (
            f"ordinal_cqr_alpha{alpha}_{ordinal_config_hash[:12]}_evaluation_metrics.json"
        )
        save_ordinal_cqr_evaluation_metrics(evaluation_path, evaluation_payload)
        lgr_logger.info(f"OrdinalCQR evaluation metrics saved to {evaluation_path}.")
        results["ordinal_cqr"] = trainer.predict(wrappers["ordinal_cqr"], test_loader)

    if "lp" in wrappers:
        results["lp"] = trainer.predict(wrappers["lp"], test_loader)

    # Save Results ---
    lgr_logger.info("Saving results to CSV...")

    for method, preds in results.items():
        path = os.path.join(
            cfg.uc.csv_path, f"{method}_alpha{alpha}_result_testset.csv"
        )
        for i, batch_res in enumerate(preds):
            save_batch_to_csv(path, batch_res, header_written=(i > 0))

    lgr_logger.info("Done.")


if __name__ == "__main__":
    run_uc_cal()
