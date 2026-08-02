import lightning as L
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

class OrdinalTargetDataset(torch.utils.data.Dataset):
    """Adapt class-only samples to the canonical ``(X, Z, Y_ord)`` contract."""

    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        x = x.unsqueeze(1) # Add dummy time dimension (C, T, H, W)
        return x, y, y

class RetinaMNISTDataModule(L.LightningDataModule):
    """
    LightningDataModule for Retina-MNIST dataset.
    Retina-MNIST contains 28x28 RGB images with 5 ordinal classes.
    """
    def __init__(self, data_dir: str = "/mnt/storage/medmnist", batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_flag = 'retinamnist'
        
        self.info = INFO[self.data_flag]
        self.DataClass = getattr(medmnist, self.info['python_class'])
        
        # Standard MedMNIST transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # MedMNIST targets are returned as 1D arrays like [3]. We extract the scalar index.
        self.target_transform = lambda x: int(x[0])

    def prepare_data(self):
        # Downloads data
        self.DataClass(split='train', download=True, root=self.data_dir)
        self.DataClass(split='val', download=True, root=self.data_dir)
        self.DataClass(split='test', download=True, root=self.data_dir)

    def setup(self, stage=None):
        if stage in ('fit', 'calibrate', None):
            full_train = self.DataClass(
                split='train', transform=self.transform, target_transform=self.target_transform, root=self.data_dir
            )
            # Stratified split 70/30
            from sklearn.model_selection import train_test_split
            import numpy as np
            
            targets = [full_train[i][1] for i in range(len(full_train))]
            indices = np.arange(len(full_train))
            train_idx, cal_idx = train_test_split(indices, test_size=0.30, stratify=targets, random_state=42)
            
            self.train_dataset = OrdinalTargetDataset(torch.utils.data.Subset(full_train, train_idx))
            self.cal_dataset = OrdinalTargetDataset(torch.utils.data.Subset(full_train, cal_idx))
            
            self.val_dataset = OrdinalTargetDataset(self.DataClass(
                split='val', transform=self.transform, target_transform=self.target_transform, root=self.data_dir
            ))
            
        if stage in ('test', None):
            self.test_dataset = OrdinalTargetDataset(self.DataClass(
                split='test', transform=self.transform, target_transform=self.target_transform, root=self.data_dir
            ))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def cal_dataloader(self):
        # Dedicated loader for Conformal Prediction Calibration step
        return DataLoader(self.cal_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
