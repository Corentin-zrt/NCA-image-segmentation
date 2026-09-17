from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_DIR = ARTIFACTS_DIR / "dataset"
MODEL_PATH = ARTIFACTS_DIR / "nca_model.npz"


@dataclass
class DatasetConfig:
    image_size: int = 96
    samples: int = 24
    shape: str = "mixed"
    seed: int = 7
