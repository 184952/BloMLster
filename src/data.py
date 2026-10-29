"""Datasett, transformasjoner og dataloadere. Importer i notebooks med:

    from src.data import CustomImageDataset, train_tf, eval_tf, get_dataloaders
"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .config import DATA_DIR

FLOWERS_DIR = DATA_DIR / "flowers"
LABELS_CSV = DATA_DIR / "labels.csv"

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet

# augmentering på trening for å motvirke overfitting
train_tf = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),   # uint8 0–255 -> float 0–1
    v2.Normalize(MEAN, STD),
])
eval_tf = v2.Compose([
    v2.Resize(256, antialias=True),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, split, transform=None, target_transform=None):
        df = pd.read_csv(annotations_file)
        self.img_labels = df[df["split"] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # torchvision fra AMD (ROCm) mangler JPEG-støtte, så bilder leses med PIL
        image = v2.functional.pil_to_tensor(Image.open(img_path).convert("RGB"))  # uint8 [3, H, W]
        label = int(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_classes(img_dir=FLOWERS_DIR):
    """Klassenavn i samme rekkefølge som labels i labels.csv (alfabetisk mappenavn)."""
    return sorted(d.name for d in img_dir.iterdir() if d.is_dir())


def get_dataloaders(batch_size=64, annotations_file=LABELS_CSV, img_dir=FLOWERS_DIR):
    """Returner (train, val, test)-dataloadere med samme oppsett som i treningen."""
    training_data = CustomImageDataset(annotations_file, img_dir, "train", train_tf)
    val_data = CustomImageDataset(annotations_file, img_dir, "val", eval_tf)
    test_data = CustomImageDataset(annotations_file, img_dir, "test", eval_tf)
    return (DataLoader(training_data, batch_size=batch_size, shuffle=True),
            DataLoader(val_data, batch_size=batch_size),
            DataLoader(test_data, batch_size=batch_size))
