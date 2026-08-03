import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch


class UTKFaceDataset(Dataset):
    """
    UTKFace Dataset for Age Estimation.
    Filenames are formatted as: [age]_[gender]_[race]_[date].jpg
    Emits canonical ``(X, Z, Y_ord)`` batches. For continuous mode, ``Z``
    is exact age; for ordinal mode, ``Z = Y_ord``.
    """

    def __init__(
        self,
        data_dir: str,
        image_paths: list[str],
        thresholds: list[float],
        label_type: str = "ordinal",
        transform=None,
    ):
        """
        Args:
            data_dir: Base directory containing the images.
            image_paths: List of relative image filenames (e.g., '26_1_0_20170116.jpg').
            thresholds: List of continuous domain boundaries to separate classes.
            label_type: "ordinal" (returns discrete class integer) or "continuous" (returns float age).
            transform: PyTorch vision transforms.
        """
        self.data_dir = Path(data_dir)
        self.image_paths = image_paths
        self.thresholds = thresholds
        self.label_type = label_type
        self.transform = transform

    def _get_class_idx(self, age: float) -> int:
        """Maps continuous age to discrete ordinal bin based on provided thresholds."""
        if age < self.thresholds[0]:
            return 0
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= age < self.thresholds[i + 1]:
                return i + 1
        return len(self.thresholds)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = self.data_dir / img_name
        
        # Parse exact continuous age from filename (e.g., '26_1_0_...jpg' -> 26.0)
        age = float(img_name.split("_")[0])
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        class_idx = self._get_class_idx(age)
        if self.label_type == "continuous":
            target = torch.tensor(age, dtype=torch.float32)
        elif self.label_type == "ordinal":
            target = torch.tensor(class_idx, dtype=torch.long)
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")

        # Add dummy time dimension (C, T, H, W) to match model expectations
        image = image.unsqueeze(1)

        return image, target, torch.tensor(class_idx, dtype=torch.long)
