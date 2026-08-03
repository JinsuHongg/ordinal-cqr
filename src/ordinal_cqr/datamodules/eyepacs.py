import os
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms

from ordinal_cqr.datasets.eyepacs import EyePACSDataset
from loguru import logger as lgr_logger

class EyePACSDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/mnt/storage/data/eyepacs/train",
        csv_path: str = "/mnt/storage/data/eyepacs/trainLabels.csv",
        batch_size: int = 256,
        num_workers: int = 4,
        label_type: str = "ordinal",
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
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

        # Build weights for WeightedRandomSampler
        class_counts = [0] * 5
        for y in train_y:
            class_counts[y] += 1
        class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
        sample_weights = [class_weights[y] for y in train_y]

        self.train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
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
