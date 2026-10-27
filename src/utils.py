"""Små hjelpefunksjoner som brukes på tvers av notebooks."""
import random

import numpy as np

from .config import FIGURES_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    """Sett seed for reproduserbare resultater."""
    random.seed(seed)
    np.random.seed(seed)


def get_device():
    """Velg GPU hvis tilgjengelig (ROCm/AMD eller CUDA), ellers CPU.

    På ROCm for Windows feiler MIOpen sin BatchNorm (kan ikke kompilere kjernen),
    så BatchNorm rutes til PyTorch sin egen kjerne. Conv o.l. bruker fortsatt MIOpen.
    """
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Velg GPU-en med mest minne, så den integrerte grafikken ikke blir valgt
    idx = max(range(torch.cuda.device_count()),
              key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    torch.cuda.set_device(idx)

    if torch.version.hip and not getattr(F.batch_norm, "_uten_miopen", False):
        _batch_norm = F.batch_norm

        def batch_norm(*args, **kwargs):
            with torch.backends.cudnn.flags(enabled=False):
                return _batch_norm(*args, **kwargs)

        batch_norm._uten_miopen = True
        F.batch_norm = batch_norm

    return torch.device(f"cuda:{idx}")


def save_fig(fig, name: str, dpi: int = 150) -> None:
    """Lagre en matplotlib-figur til reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
