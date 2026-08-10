import os
import csv
import hydra
from loguru import logger as lgr_logger
import torch
# PyTorch 2.6 security monkey-patch for Hydra dict configs
_original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load
import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from ordinal_cqr.datamodules import FlareSuryaBenchDataModule
from ordinal_cqr.explainability import (
    OrdinalAPSWrapper,
    MinCPSWrapper,
    MinRCPSWrapper,
    COPOCWrapper,
    BinomialLACWrapper,
    RiskControlWrapper,
)
from ordinal_cqr.models import ResNet18COPOC, ResNetCls


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
    config_name="CLS_resnet18_calibration_surya_bench.yaml",
    version_base=None,
)
def run_ordinal_uc_cal(cfg):
    L.seed_everything(cfg.get("seed", 42), workers=True)
    methods = cfg.uc.get("methods", ["oaps", "min_cps", "min_rcps", "copoc", "risk_control"])
    if cfg.data.repo == "retinamnist":
        from ordinal_cqr.datamodules.retina_mnist import RetinaMNISTDataModule
        datamodule = RetinaMNISTDataModule(data_dir="/mnt/storage/medmnist", batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    elif cfg.data.repo == "adience":
        from ordinal_cqr.datamodules.adience import AdienceDataModule
        datamodule = AdienceDataModule(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, label_type='ordinal')
    elif cfg.data.repo == "utkface":
        from ordinal_cqr.datamodules.utkface import UTKFaceDataModule
        datamodule = UTKFaceDataModule(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, label_type=cfg.data.get('label_type', 'ordinal'))
    elif cfg.data.repo == "eyepacs":
        from ordinal_cqr.datamodules.eyepacs import EyePACSDataModule
        datamodule = EyePACSDataModule(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, label_type=cfg.data.get('label_type', 'ordinal'))
    else:
        datamodule = FlareSuryaBenchDataModule(cfg=cfg)
    datamodule.setup(stage="calibrate")
    datamodule.setup(stage="test")

    calibration_loader = datamodule.cal_dataloader()
    test_loader = datamodule.test_dataloader()

    # Load Model
    ckpt_name = cfg.check_point.get(
        "resnet18_copoc_cls",
        cfg.check_point.get("resnet18_binomial_cls", cfg.check_point.get("resnet18_cls", "")),
    )
    cls_pretrained_path = os.path.join(
        cfg.check_point.base, ckpt_name
    )
    model = ResNetCls.load_from_checkpoint(
        cls_pretrained_path, strict=False, weights_only=False
    )
    if "copoc" in methods and not isinstance(model.base_model, ResNet18COPOC):
        raise RuntimeError(
            "Method 'copoc' requires a fresh resnet18_copoc checkpoint; Binomial "
            "and standard ResNet checkpoints are separate baselines."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wrappers = {}
    for method_name in methods:
        lgr_logger.info(f"Initializing and calibrating Ordinal UQ method: {method_name}")

        alpha = cfg.uc.significance_level
        num_classes = cfg.uc.num_classes
        class_mapping = cfg.uc.get(
            "class_mapping", {"FQ": 0, "B": 1, "C": 2, "M": 3, "X": 4}
        )

        wrapper = None

        match method_name:
            case "oaps":
                class_wise = cfg.uc.get("class_wise", False)
                wrapper = OrdinalAPSWrapper(
                    trained_model=model,
                    num_classes=num_classes,
                    alpha=alpha,
                    class_wise=class_wise,
                    class_mapping=class_mapping,
                )
            case "min_cps":
                wrapper = MinCPSWrapper(
                    trained_model=model,
                    num_classes=num_classes,
                    alpha=alpha,
                    class_mapping=class_mapping,
                )
            case "min_rcps":
                wrapper = MinRCPSWrapper(
                    trained_model=model,
                    num_classes=num_classes,
                    alpha=alpha,
                    reg_weight=cfg.uc.get("reg_weight", 0.01),
                    class_mapping=class_mapping,
                )
            case "copoc":
                wrapper = COPOCWrapper(
                    trained_model=model,
                    num_classes=num_classes,
                    alpha=alpha,
                    class_mapping=class_mapping,
                    numerical_tolerance=cfg.uc.get("numerical_tolerance", 1e-5),
                )
            case "binomial_lac":
                wrapper = BinomialLACWrapper(
                    trained_model=model,
                    num_classes=num_classes,
                    alpha=alpha,
                    class_mapping=class_mapping,
                    numerical_tolerance=cfg.uc.get("numerical_tolerance", 1e-5),
                )
            case "risk_control":
                wrapper = RiskControlWrapper(
                    trained_model=model,
                    num_classes=num_classes,
                    alpha=alpha,
                    class_mapping=class_mapping,
                    delta=cfg.uc.get("delta", 0.1),
                )
            case _:
                raise ValueError(f"Unknown method: {method_name}")

        # Calibration
        wrapper.to(device)
        wrapper.calibrate(calibration_loader)
        wrappers[method_name] = wrapper

    # Loggers
    loggers = []
    if cfg.get("wandb"):
        wandb_logger = WandbLogger(
            project=cfg.wandb.get("project", "default_project"),
            entity=cfg.wandb.get("entity", "default_entity"),
            name=f"calibration_ordinal_run_{cfg.experiment.task}",
            save_dir=cfg.wandb.get("save_dir", "./wandb_logs"),
        )
        loggers.append(wandb_logger)

    csv_logger = CSVLogger(save_dir=cfg.uc.csv_path, name="summary_ordinal")
    loggers.append(csv_logger)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=loggers,
    )

    lgr_logger.info("Running Evaluation on Test Set...")
    results = {}

    for method_name, wrapper in wrappers.items():
        lgr_logger.info(f"Running Metrics Evaluation for {method_name}...")
        trainer.test(wrapper, test_loader)

        lgr_logger.info(f"Running Prediction for {method_name}...")
        results[method_name] = trainer.predict(wrapper, test_loader)

    lgr_logger.info("Saving results to CSV...")
    for method, preds in results.items():
        path = os.path.join(
            cfg.uc.csv_path,
            f"{method}_ordinal_alpha{cfg.uc.significance_level}_result_testset.csv",
        )
        for i, batch_res in enumerate(preds):
            save_batch_to_csv(path, batch_res, header_written=(i > 0))

    lgr_logger.info("All Ordinal UQ methods processed.")


if __name__ == "__main__":
    run_ordinal_uc_cal()
