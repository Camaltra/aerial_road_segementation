import albumentations as A
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms
import os


class RoadDataset(Dataset):
    def __init__(self, dataset_type: str, transform: None | A.Compose = None) -> None:
        self.items = [
            img_path.split("/")[-1]
            for img_path in glob(f"dataset/{dataset_type}/img/*.tif")
        ]
        self.path = f"dataset/{dataset_type}"
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix: int) -> tuple[np.ndarray, np.ndarray]:
        img = np.array(Image.open(f"{self.path}/img/{self.items[ix]}").convert("RGB"))
        mask = np.array(Image.open(f"{self.path}/mask/{self.items[ix]}").convert("L"))

        if transforms is not None:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]

        return img, mask
