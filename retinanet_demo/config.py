"""Configuration for retinanet-demo."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for retinanet-demo."""

    data_directory: Path = Path("/code/data")

    # Dataset Paths
    dataset_directory: Path = data_directory / "dataset"
    images_directory: Path = dataset_directory / "images"
    annotations_directory: Path = dataset_directory / "annotations"

    # Model Paths
    models_directory: Path = data_directory / "models"
    model_path: Path = models_directory / "model.pt"

    # Metrics Path
    metrics_path: Path = data_directory / "metrics.json"

    labels: tuple[str, ...] = (
        "bicycle",
        "bird",
        "car",
        "chair",
        "dog",
        "person",
        "truck",
    )

    # Hyperparameters
    batch_size: int = 4
    learning_rate: float = 1e-3
    max_epochs: int = 10
    num_workers: int = 4
