# import argparse
import os
import hashlib
import copy
import json
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import hydra
from loguru import logger as lgr_logger
from omegaconf import OmegaConf
import omegaconf

import torch
import warnings

# PyTorch 2.6 changed torch.load to weights_only=True by default.
# This breaks PyTorch Lightning checkpoint loading for Hydra config dicts.
# We globally patch torch.load to bypass this local security restriction.
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    else:
        # Some calls might not pass it as a kwarg but rely on default
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import lightning as L
from lightning.pytorch import Trainer

from ordinal_cqr.datamodules import (
    FlareSuryaBenchDataModule,
)
from ordinal_cqr.models import ResNetMCD, ResNetQR, ResNetCls
from ordinal_cqr.utils import build_wandb, build_callbacks

torch.set_float32_matmul_precision("medium")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    """Write strict JSON without exposing a partially written status file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _resolved_config_payload(cfg) -> tuple[dict, str, str]:
    payload = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    protocol = copy.deepcopy(payload)
    protocol.get("model", {}).pop("save_ckpt_path", None)
    protocol_canonical = json.dumps(
        protocol, sort_keys=True, separators=(",", ":")
    )
    return (
        payload,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        hashlib.sha256(protocol_canonical.encode("utf-8")).hexdigest(),
    )


def _git_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    ignored_runtime_prefixes = ("outputs/", "results/", "logs/", "assets/")
    dirty = any(
        not (
            line.startswith("?? ")
            and (
                line[3:].startswith(ignored_runtime_prefixes)
                or line[3:].startswith("slurm-")
                or line[3:].endswith((".log", "_exit_code.txt"))
            )
        )
        for line in status
    )
    return {"code_commit": commit, "git_dirty": dirty}


def _initialize_training_run(cfg) -> tuple[Path | None, dict | None, float]:
    """Persist resolved Surya training identity when launched by the Slurm wrapper."""
    started = time.time()
    value = os.getenv("SURYA_RUN_DIR")
    if not value:
        return None, None, started
    run_dir = Path(value)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved, config_hash, protocol_hash = _resolved_config_payload(cfg)
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    split_audit = run_dir / "split_audit.json"
    provenance = {
        "schema_version": "surya-training-run-v1",
        "status": "started",
        "dataset": "solar_flare",
        "dataset_contract_version": "0.3.0",
        "method": os.getenv("SURYA_METHOD", str(cfg.model.module_type)),
        "seed": int(cfg.seed),
        "configuration_sha256": config_hash,
        "protocol_configuration_sha256": protocol_hash,
        "resolved_configuration": resolved,
        **_git_metadata(),
        "checkpoint_selection_criterion": "validation_pinball_loss"
        if cfg.model.module_type == "qr"
        else "validation_cross_entropy",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "lightning_version": L.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "started_at": _utc_now(),
        "split_audit": {
            "path": str(split_audit),
            "sha256": hashlib.sha256(split_audit.read_bytes()).hexdigest(),
        }
        if split_audit.is_file()
        else None,
    }
    _atomic_json(run_dir / "provenance.json", provenance)
    _atomic_json(
        run_dir / "run_status.json",
        {"status": "started", "stage": "training", "updated_at": _utc_now()},
    )
    return run_dir, provenance, started


def _finish_training_run(
    run_dir: Path | None,
    provenance: dict | None,
    started: float,
    *,
    trainer: Trainer | None = None,
    error: Exception | None = None,
) -> None:
    if run_dir is None or provenance is None:
        return
    if error is not None:
        provenance.update(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "runtime_seconds": time.time() - started,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        )
        status = {
            "status": "failed",
            "stage": "training",
            "exception_type": type(error).__name__,
            "message": str(error),
            "updated_at": _utc_now(),
        }
    else:
        checkpoint = trainer.checkpoint_callback
        best_score = checkpoint.best_model_score
        provenance.update(
            {
                "status": "training_complete",
                "completed_at": _utc_now(),
                "runtime_seconds": time.time() - started,
                "selected_checkpoint": checkpoint.best_model_path,
                "selected_validation_loss": float(best_score)
                if best_score is not None
                else None,
                "last_checkpoint": checkpoint.last_model_path,
            }
        )
        status = {
            "status": "training_complete",
            "stage": "complete",
            "updated_at": _utc_now(),
        }
    _atomic_json(run_dir / "provenance.json", provenance)
    _atomic_json(run_dir / "run_status.json", status)


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)
    lgr_logger.info(f"Loaded config from {config_path}")
    return cfg


def build_model(cfg):
    module_type = cfg.model.module_type

    if module_type == "mcd":
        return ResNetMCD(
            model_type=cfg.model.type,
            module_dict=cfg.model.get(cfg.model.module_type),
            base_model_dict=cfg.model.get(cfg.model.type),
            loss_type=cfg.model.loss.type,
            optimizer_dict=cfg.optimizer,
            scheduler_dict=cfg.scheduler,
        )

    elif module_type == "qr":
        return ResNetQR(
            model_type=cfg.model.type,
            module_dict=cfg.model.get(cfg.model.module_type),
            base_model_dict=cfg.model.get(cfg.model.type),
            optimizer_dict=cfg.optimizer,
            scheduler_dict=cfg.scheduler,
        )

    elif module_type == "cls":
        return ResNetCls(
            model_type=cfg.model.type,
            base_model_dict=cfg.model.get(cfg.model.type),
            optimizer_dict=cfg.optimizer,
            scheduler_dict=cfg.scheduler,
            loss_dict=cfg.model.get("loss"),
        )


def _fit(cfg) -> Trainer:
    # Datamodule
    if cfg.data.repo == "retinamnist":
        from ordinal_cqr.datamodules.retina_mnist import RetinaMNISTDataModule
        datamodule = RetinaMNISTDataModule(data_dir="/mnt/storage/medmnist", batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    elif cfg.data.repo == "adience":
        from ordinal_cqr.datamodules.adience import AdienceDataModule
        # Use continuous label for QR models to predict actual ages
        label_type = 'continuous' if cfg.model.module_type == 'qr' else 'ordinal'
        datamodule = AdienceDataModule(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, label_type=label_type)
    elif cfg.data.repo == "utkface":
        from ordinal_cqr.datamodules.utkface import UTKFaceDataModule
        datamodule = UTKFaceDataModule(
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            label_type=getattr(cfg.data, "label_type", "ordinal"),
        )
    elif cfg.data.repo == "eyepacs":
        from ordinal_cqr.datamodules.eyepacs import EyePACSDataModule
        datamodule = EyePACSDataModule(
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            label_type=getattr(cfg.data, "label_type", "ordinal"),
            sampling_strategy=getattr(
                cfg.data, "sampling_strategy", "inverse_frequency"
            ),
        )
    else:
        datamodule = FlareSuryaBenchDataModule(cfg=cfg)

    # Load model
    model = build_model(cfg=cfg)

    # Create wandb obejct
    wandb_logger = build_wandb(cfg=cfg)

    # Trainer
    callbacks = build_callbacks(cfg=cfg, wandb_logger=wandb_logger)
    trainer = Trainer(
        enable_progress_bar=False,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        strategy=cfg.trainer.strategy,
        deterministic=cfg.trainer.get("deterministic", False),
    )

    lgr_logger.info(f"Start training...")
    ckpt = (
        os.path.join(cfg.model.save_ckpt_path, cfg.model.ckpt)
        if cfg.model.ckpt
        else None
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
    return trainer


@hydra.main(
    config_path="../../configs",
    config_name="QR_resnet18_train_surya_bench",
    version_base=None,
)
def train(cfg):
    L.seed_everything(int(cfg.seed), workers=True)
    run_dir, provenance, started = _initialize_training_run(cfg)
    try:
        trainer = _fit(cfg)
    except Exception as error:
        _finish_training_run(
            run_dir, provenance, started, error=error
        )
        raise
    _finish_training_run(run_dir, provenance, started, trainer=trainer)


if __name__ == "__main__":
    train()
