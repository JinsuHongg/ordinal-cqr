import datetime
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger


def build_wandb(cfg):
    """Build a WandbLogger when wandb is configured and enabled, else return None."""
    wandb_cfg = cfg.get("wandb")
    if not wandb_cfg or not wandb_cfg.get("enabled", True):
        return None

    name = f"{cfg.model.type}_lr{cfg.optimizer.lr}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return WandbLogger(
        entity=wandb_cfg.get("entity", None),
        project=wandb_cfg.get("project", "ordinal-cqr"),
        save_dir=wandb_cfg.get("save_dir", "./assets"),
        offline=wandb_cfg.get("offline", True),
        log_model=wandb_cfg.get("log_model", False),
        save_code=wandb_cfg.get("save_code", False),
        notes=wandb_cfg.get("notes", ""),
        tags=wandb_cfg.get("tag", None),
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
