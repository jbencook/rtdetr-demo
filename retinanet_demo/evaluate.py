"""Evaluate the model."""
import pytorch_lightning as pl
import torch

from .config import Config
from .train import Lightning


def evaluate_model() -> None:
    """Evaluate the model."""
    lightning = Lightning()
    lightning.model.load_state_dict(torch.load(Config.model_path))
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.test(lightning)
