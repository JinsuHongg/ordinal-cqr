# import argparse
import os
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


@hydra.main(
    config_path="../../configs",
    config_name="QR_resnet18_train_surya_bench",
    version_base=None,
)
def train(cfg):
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
    )

    lgr_logger.info(f"Start training...")
    ckpt = (
        os.path.join(cfg.model.save_ckpt_path, cfg.model.ckpt)
        if cfg.model.ckpt
        else None
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
    # trainer.test(dataloaders=datamodule)


if __name__ == "__main__":
    train()
