"""Configuration for retinanet-demo."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for retinanet-demo."""

    data_directory: Path = Path("/code/data")
    dataset_directory: Path = data_directory / "dataset"
    models_directory: Path = data_directory / "models"
    images_directory: Path = dataset_directory / "images"
    annotations_directory: Path = dataset_directory / "annotations"
