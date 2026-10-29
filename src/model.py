"""Modellarkitektur og lasting av lagret modell. Importer i notebooks med:

    from src.model import FlowerNet, load_model
"""
import torch
from torch import nn

from .config import MODELS_DIR


class FlowerNet(nn.Module):
    """Enkel CNN: fire konvolusjonsblokker (224 -> 14) og et lineært lag."""

    def __init__(self, num_classes):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))  # 224 -> 14
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path=MODELS_DIR / "flowernet.pt", device="cpu"):
    """Last modell + metadata (classes, mean, std, img_size, ...) fra torch.save-dicten."""
    ckpt = torch.load(path, map_location=device)
    model = FlowerNet(len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
