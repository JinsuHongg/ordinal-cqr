import os
import pandas as pd
import lightning as L
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms

from ordinal_cqr.datasets.eyepacs import EyePACSDataset
from loguru import logger as lgr_logger


def build_eyepacs_sample_weights(
    labels: list[int], sampling_strategy: str
) -> torch.Tensor | None:
    """Return per-sample weights for a declared EyePACS sampling strategy."""
    if sampling_strategy == "natural":
        return None
    exponents = {
        "sqrt_inverse_frequency": -0.5,
        "inverse_frequency": -1.0,
    }
    if sampling_strategy not in exponents:
        raise ValueError(
            "sampling_strategy must be 'natural', 'sqrt_inverse_frequency', "
            "or 'inverse_frequency'."
        )
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    if label_tensor.numel() == 0:
        raise ValueError("EyePACS training labels must not be empty.")
    class_counts = torch.bincount(label_tensor, minlength=5).to(dtype=torch.float64)
    if torch.any(class_counts == 0):
        raise ValueError("Every EyePACS class must be represented in the training split.")
    class_weights = class_counts.pow(exponents[sampling_strategy])
    return class_weights.gather(0, label_tensor)

class EyePACSDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/mnt/storage/data/eyepacs/train",
        csv_path: str = "/mnt/storage/data/eyepacs/trainLabels.csv",
        batch_size: int = 256,
        num_workers: int = 4,
        label_type: str = "ordinal",
        random_seed: int = 42,
        sampling_strategy: str = "inverse_frequency",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_type = label_type
        self.random_seed = random_seed
        self.sampling_strategy = sampling_strategy

        # Setup image transforms
        self.transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage: str = None):
        lgr_logger.info("Setting up EyePACS dataset...")

        df = pd.read_csv(self.csv_path)
        
        image_paths = []
        labels = []
        for _, row in df.iterrows():
            img = str(row['image'])
            if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                img += '.jpeg'
            image_paths.append(img)
            labels.append(int(row['level']))

        lgr_logger.info(f"Found {len(image_paths)} EyePACS samples.")

        # Split: 60% Train, 10% Val, 20% Cal, 10% Test
        # 1. Split Train vs (Val+Cal+Test) -> 60% / 40%
        train_img, temp_img, train_y, temp_y = train_test_split(
            image_paths, labels, test_size=0.4, stratify=labels, random_state=self.random_seed
        )

        # 2. Split Temp into Val (25%), Cal (50%), Test (25%) relative to the 40% chunk
        val_img, rem_img, val_y, rem_y = train_test_split(
            temp_img, temp_y, test_size=0.75, stratify=temp_y, random_state=self.random_seed
        )

        cal_img, test_img, cal_y, test_y = train_test_split(
            rem_img, rem_y, test_size=0.3333, stratify=rem_y, random_state=self.random_seed
        )

        self.train_dataset = EyePACSDataset(
            self.data_dir, train_img, train_y, self.label_type, self.transform_train
        )
        self.val_dataset = EyePACSDataset(
            self.data_dir, val_img, val_y, self.label_type, self.transform_val
        )
        self.cal_dataset = EyePACSDataset(
            self.data_dir, cal_img, cal_y, self.label_type, self.transform_val
        )
        self.test_dataset = EyePACSDataset(
            self.data_dir, test_img, test_y, self.label_type, self.transform_val
        )

        sample_weights = build_eyepacs_sample_weights(
            train_y, self.sampling_strategy
        )
        self.train_sampler = (
            None
            if sample_weights is None
            else WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )

    def cal_dataloader(self):
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )
