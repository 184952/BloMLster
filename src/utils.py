"""Små hjelpefunksjoner som brukes på tvers av notebooks."""
import random

import numpy as np

from .config import FIGURES_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    """Sett seed for reproduserbare resultater."""
    random.seed(seed)
    np.random.seed(seed)


def save_fig(fig, name: str, dpi: int = 150) -> None:
    """Lagre en matplotlib-figur til reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
