"""Felles stier og innstillinger. Importer i notebooks med:

    import sys; sys.path.append("..")
    from src.config import RAW_DIR, PROCESSED_DIR, SEED
"""
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

SEED = int(os.getenv("RANDOM_SEED", 42))
