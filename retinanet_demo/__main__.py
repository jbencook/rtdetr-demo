"""Expose the CLI."""
import fire

from .dataset import version_annotations, version_images
from .train import save_checkpoint, train_model


def main() -> None:
    """Call CLI commands."""
    fire.Fire(
        {
            "version-annotations": version_annotations,
            "version-images": version_images,
            "save-checkpoint": save_checkpoint,
            "train-model": train_model,
        }
    )
