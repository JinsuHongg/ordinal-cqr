import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class EyePACSDataset(Dataset):
    """
    EyePACS Dataset for Diabetic Retinopathy.
    Emits canonical ``(X, Z, Y_ord)`` batches. EyePACS has no separate numeric
    measurement, so continuous mode uses the floating class-index embedding.
    """
    def __init__(
        self,
        data_dir: str,
        image_paths: list[str],
        labels: list[int],
        label_type: str = "ordinal",
        transform=None,
    ):
        """
        Args:
            data_dir: Base directory containing the images.
            image_paths: List of relative image filenames (e.g., '10_left.jpeg').
            labels: List of integer labels corresponding to the images.
            label_type: "ordinal" (returns discrete class integer) or "continuous" (returns float).
            transform: PyTorch vision transforms.
        """
        self.data_dir = Path(data_dir)
        self.image_paths = image_paths
        self.labels = labels
        self.label_type = label_type
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = self.data_dir / img_name
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.label_type == "continuous":
            target = torch.tensor(float(label), dtype=torch.float32)
        elif self.label_type == "ordinal":
            target = torch.tensor(label, dtype=torch.long)
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")

        # Add dummy time dimension (C, T, H, W) to match model expectations
        image = image.unsqueeze(1)

        return image, target, torch.tensor(label, dtype=torch.long)
