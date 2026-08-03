import os
import torch
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from ..datasets.adience import AdienceDataset
from loguru import logger as lgr_logger

VALID_AGES = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
AGE_TO_ORDINAL = {k: v for v, k in enumerate(VALID_AGES)}
AGE_TO_CONTINUOUS = {
    '(0, 2)': 1.0,
    '(4, 6)': 5.0,
    '(8, 12)': 10.0,
    '(15, 20)': 17.5,
    '(25, 32)': 28.5,
    '(38, 43)': 40.5,
    '(48, 53)': 50.5,
    '(60, 100)': 65.0
}

class AdienceDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/mnt/storage/medmnist/adience/AdienceBenchmarkGenderAndAgeClassification",
        batch_size: int = 32,
        label_type: str = 'ordinal',
        num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.label_type = label_type
        self.num_workers = num_workers
        
        # Standard ResNet augmentations
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        lgr_logger.info("Setting up Adience dataset...")
        dfs = []
        for fold in range(5):
            fold_file = os.path.join(self.data_dir, f"fold_{fold}_data.txt")
            if not os.path.exists(fold_file):
                raise FileNotFoundError(f"Missing fold file: {fold_file}")
                
            df = pd.read_csv(fold_file, sep='\t')
            dfs.append(df)
            
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Clean and filter to strictly 8 canonical classes
        full_df = full_df.dropna(subset=['age', 'user_id', 'face_id', 'original_image'])
        
        # Some labels have trailing spaces or formatting issues
        full_df['age'] = full_df['age'].astype(str).str.strip()
        full_df = full_df[full_df['age'].isin(VALID_AGES)].copy()
        
        # Construct relative image path: faces/user_id/coarse_tilt_aligned_face.face_id.original_image
        full_df['image_path'] = full_df.apply(
            lambda r: f"faces/{r['user_id']}/coarse_tilt_aligned_face.{r['face_id']}.{r['original_image']}", axis=1
        )
        
        # Map label to ordinal integer and continuous float
        full_df['label'] = full_df['age'].map(AGE_TO_ORDINAL)
        full_df['continuous_label'] = full_df['age'].map(AGE_TO_CONTINUOUS)
        
        # Drop rows where image file does not actually exist on disk to prevent crash
        valid_rows = []
        for idx, row in full_df.iterrows():
            if os.path.exists(os.path.join(self.data_dir, row['image_path'])):
                valid_rows.append(row)
        full_df = pd.DataFrame(valid_rows)
        
        lgr_logger.info(f"Loaded {len(full_df)} valid Adience samples matching canonical age brackets.")
        
        # Splits: 60% Train, 20% Cal, 10% Val, 10% Test
        train_cal, val_test = train_test_split(full_df, test_size=0.20, stratify=full_df['label'], random_state=42)
        self.val_df, self.test_df = train_test_split(val_test, test_size=0.50, stratify=val_test['label'], random_state=42)
        
        self.train_df, self.cal_df = train_test_split(train_cal, test_size=0.25, stratify=train_cal['label'], random_state=42) # 0.25 * 0.8 = 0.2
        
        self.train_dataset = AdienceDataset(self.train_df, self.data_dir, transform=self.transform_train, label_type=self.label_type)
        self.val_dataset = AdienceDataset(self.val_df, self.data_dir, transform=self.transform_eval, label_type=self.label_type)
        self.cal_dataset = AdienceDataset(self.cal_df, self.data_dir, transform=self.transform_eval, label_type=self.label_type)
        self.test_dataset = AdienceDataset(self.test_df, self.data_dir, transform=self.transform_eval, label_type=self.label_type)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def cal_dataloader(self):
        return DataLoader(self.cal_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
