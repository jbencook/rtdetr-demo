"""Evaluate the model."""
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch

from .config import Config
from .train import Lightning


def evaluate_model(checkpoint_path: Optional[Union[str, Path]] = None) -> None:
    """Evaluate the model."""
    if checkpoint_path:
        lightning = Lightning.load_from_checkpoint(checkpoint_path)
    else:
        lightning = Lightning()
        lightning.model.load_state_dict(torch.load(Config.model_path))
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.test(lightning)
