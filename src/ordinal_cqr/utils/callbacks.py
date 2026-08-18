from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


def build_callbacks(cfg, wandb_logger):

    if cfg.model.module_type == "qr":
        quantiles = cfg.model.qr.quantiles
        cf_level = int((quantiles[2] - quantiles[0]) * 100)
        ckpt_name = (
            f"{wandb_logger.experiment.id}_"
            f"{cfg['experiment']['ckpt_file_name']}_q{cf_level}_"
            "{epoch}-{val_loss:.4f}"
        )
    else:
        ckpt_name = (
            f"{wandb_logger.experiment.id}_"
            f"{cfg['experiment']['ckpt_file_name']}_"
            "{epoch}-{val_loss:.4f}"
        )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor=cfg["scheduler"]["monitor"],
            dirpath=cfg["model"]["save_ckpt_path"],
            filename=ckpt_name,
            save_top_k=3,
            save_last=True,
            verbose=True,
            mode="min",
        ),
    ]
    early_stopping = cfg.trainer.get("early_stopping")
    if early_stopping and early_stopping.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping.get("monitor", cfg["scheduler"]["monitor"]),
                mode=early_stopping.get("mode", "min"),
                patience=int(early_stopping.get("patience", 3)),
                min_delta=float(early_stopping.get("min_delta", 0.0)),
            )
        )
    return callbacks
