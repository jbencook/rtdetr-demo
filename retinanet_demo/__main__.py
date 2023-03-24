"""Expose the CLI."""
import fire

from .dataset import version_annotations, version_images


def main() -> None:
    """Call CLI commands."""
    fire.Fire(
        {
            "version-annotations": version_annotations,
            "version-images": version_images,
        }
    )
