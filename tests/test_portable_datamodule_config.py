import inspect
from omegaconf import OmegaConf

from ordinal_cqr.datamodules.adience import AdienceDataModule
from ordinal_cqr.datamodules.eyepacs import EyePACSDataModule
from ordinal_cqr.datamodules.retina_mnist import RetinaMNISTDataModule
from ordinal_cqr.datamodules.utkface import UTKFaceDataModule
from ordinal_cqr.utils.callbacks import build_callbacks
from ordinal_cqr.utils.wandb_logging import build_wandb


def test_datamodule_defaults_have_no_cluster_paths() -> None:
    for cls in (
        RetinaMNISTDataModule,
        UTKFaceDataModule,
        AdienceDataModule,
        EyePACSDataModule,
    ):
        sig = inspect.signature(cls.__init__)
        for param_name in ("data_dir", "csv_path"):
            if param_name in sig.parameters:
                default = str(sig.parameters[param_name].default)
                assert "/mnt/storage" not in default
                assert "/scratch/" not in default
                assert "/anvil/" not in default


def test_wandb_can_be_disabled_and_callbacks_work_without_wandb() -> None:
    assert build_wandb(OmegaConf.create({"wandb": {"enabled": False}})) is None
    assert build_wandb(OmegaConf.create({})) is None

    cfg = OmegaConf.create(
        {
            "experiment": {"ckpt_file_name": "test_ckpt"},
            "model": {"module_type": "qr", "qr": {"quantiles": [0.05, 0.5, 0.95]}, "save_ckpt_path": "./assets/checkpoints/qr"},
            "scheduler": {"monitor": "val_loss"},
            "trainer": {"early_stopping": {"enabled": False}},
        }
    )
    callbacks = build_callbacks(cfg, wandb_logger=None)
    assert len(callbacks) >= 2
