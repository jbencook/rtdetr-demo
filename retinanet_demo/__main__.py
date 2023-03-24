"""Expose the CLI."""
import fire

from .dataset import version_annotations


def main() -> None:
    """Call CLI commands."""
    fire.Fire({"version-annotations": version_annotations})
