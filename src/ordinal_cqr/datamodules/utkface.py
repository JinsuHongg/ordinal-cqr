import os
import lightning as L
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

from ordinal_cqr.datasets.utkface import UTKFaceDataset
from loguru import logger as lgr_logger


class UTKFaceDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/mnt/storage/data/utkface/UTKFace",
        batch_size: int = 256,
        num_workers: int = 4,
        thresholds: list[float] = [20.0, 40.0, 60.0, 80.0],
        label_type: str = "ordinal",
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.thresholds = thresholds
        self.label_type = label_type
        self.random_seed = random_seed

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

    def _get_class_idx(self, age: float) -> int:
        if age < self.thresholds[0]:
            return 0
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= age < self.thresholds[i + 1]:
                return i + 1
        return len(self.thresholds)

    def setup(self, stage: str = None):
        lgr_logger.info("Setting up UTKFace dataset...")
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith(".jpg")]
        
        # Calculate stratify labels
        stratify_labels = []
        valid_files = []
        for f in all_files:
            try:
                age = float(f.split("_")[0])
                cls_idx = self._get_class_idx(age)
                stratify_labels.append(cls_idx)
                valid_files.append(f)
            except Exception:
                continue

        lgr_logger.info(f"Found {len(valid_files)} valid UTKFace samples.")

        # Split: 60% Train, 10% Val, 20% Cal, 10% Test
        # 1. Split Train vs (Val+Cal+Test) -> 60% / 40%
        train_files, temp_files, train_y, temp_y = train_test_split(
            valid_files, stratify_labels, test_size=0.4, stratify=stratify_labels, random_state=self.random_seed
        )

        # 2. Split Temp into Val (25%), Cal (50%), Test (25%) relative to the 40% chunk
        # First split Val off
        val_files, rem_files, val_y, rem_y = train_test_split(
            temp_files, temp_y, test_size=0.75, stratify=temp_y, random_state=self.random_seed
        )

        # Then split remaining (30% overall) into Cal (20% overall) and Test (10% overall)
        # So Cal gets 2/3 of remainder, Test gets 1/3
        cal_files, test_files, cal_y, test_y = train_test_split(
            rem_files, rem_y, test_size=0.3333, stratify=rem_y, random_state=self.random_seed
        )

        self.train_dataset = UTKFaceDataset(
            self.data_dir, train_files, self.thresholds, self.label_type, self.transform_train
        )
        self.val_dataset = UTKFaceDataset(
            self.data_dir, val_files, self.thresholds, self.label_type, self.transform_val
        )
        self.cal_dataset = UTKFaceDataset(
            self.data_dir, cal_files, self.thresholds, self.label_type, self.transform_val
        )
        self.test_dataset = UTKFaceDataset(
            self.data_dir, test_files, self.thresholds, self.label_type, self.transform_val
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def cal_dataloader(self):
        return DataLoader(
            self.cal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
