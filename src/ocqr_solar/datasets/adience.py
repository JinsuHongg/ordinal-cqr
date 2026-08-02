import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class AdienceDataset(Dataset):
    def __init__(self, data_df, image_dir, transform=None, label_type='ordinal'):
        """
        Adience Age Dataset
        
        Args:
            data_df (pd.DataFrame): DataFrame containing 'image_path' and 'label' columns.
            image_dir (str): Root directory containing the 'faces' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
            label_type (str): 'ordinal' returns the class index as ``Z``;
                'continuous' returns the documented age-bin representative as ``Z``.
        """
        self.data_df = data_df
        self.image_dir = image_dir
        self.transform = transform
        self.label_type = label_type

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        img_path = os.path.join(self.image_dir, row['image_path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for broken images
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        if self.transform:
            image = self.transform(image)
            
        ordinal_label = torch.tensor(int(row['label']), dtype=torch.long)
        
        if self.label_type == 'continuous':
            target = torch.tensor(float(row['continuous_label']), dtype=torch.float32)
        else:
            target = ordinal_label
            
        # Add dummy time dimension (C, T, H, W) to match model expectations
        image = image.unsqueeze(1)
        return image, target, ordinal_label
