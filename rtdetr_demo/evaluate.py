"""Evaluate the model."""
from pathlib import Path
from typing import Union

import pytorch_lightning as pl

from .train import Lightning


def evaluate_model(checkpoint_path: Union[str, Path]) -> None:
    """Evaluate the model."""
    lightning = Lightning.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.test(lightning)
