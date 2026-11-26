from .adience import AdienceDataModule
from .flare import FlareHelioviewerRegDataModule, FlareSuryaBenchDataModule
from .retina_mnist import RetinaMNISTDataModule


def build_datamodule(
    cfg,
    *,
    include_thresholds: bool = False,
    include_sampling_strategy: bool = False,
    fallback_helioviewer: bool = False,
):
    """Instantiate the configured LightningDataModule from a Hydra config."""
    repo = cfg.data.get("repo")
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers

    if repo == "retinamnist":
        return RetinaMNISTDataModule(
            data_dir=cfg.data.get("data_dir", "./data/medmnist"),
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if repo == "adience":
        default_label = (
            "continuous"
            if cfg.get("model", {}).get("module_type") == "qr"
            else "ordinal"
        )
        return AdienceDataModule(
            data_dir=cfg.data.get("data_dir", "./data/adience"),
            batch_size=batch_size,
            num_workers=num_workers,
            label_type=cfg.data.get("label_type", default_label),
        )
    if repo == "utkface":
        from .utkface import UTKFaceDataModule

        kwargs = {
            "data_dir": cfg.data.get("data_dir", "./data/utkface"),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "label_type": cfg.data.get("label_type", "ordinal"),
        }
        if include_thresholds:
            kwargs["thresholds"] = cfg.uc.get("thresholds", [20.0, 40.0, 60.0, 80.0])
        return UTKFaceDataModule(**kwargs)
    if repo == "eyepacs":
        from .eyepacs import EyePACSDataModule

        kwargs = {
            "data_dir": cfg.data.get("data_dir", "./data/eyepacs/train"),
            "csv_path": cfg.data.get("csv_path", "./data/eyepacs/trainLabels.csv"),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "label_type": cfg.data.get("label_type", "ordinal"),
        }
        if include_sampling_strategy:
            kwargs["sampling_strategy"] = cfg.data.get(
                "sampling_strategy", "inverse_frequency"
            )
        return EyePACSDataModule(**kwargs)
    if fallback_helioviewer and "input_zarr_path" not in cfg.data:
        return FlareHelioviewerRegDataModule(cfg=cfg)
    return FlareSuryaBenchDataModule(cfg=cfg)


__all__ = [
    "AdienceDataModule",
    "FlareHelioviewerRegDataModule",
    "FlareSuryaBenchDataModule",
    "RetinaMNISTDataModule",
    "build_datamodule",
]
