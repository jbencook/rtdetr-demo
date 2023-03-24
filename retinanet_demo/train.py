"""Train the model."""
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
from torchvision import models

from .config import Config
from .dataset import Dataset


class Lightning(pl.LightningModule):
    """Lightning module for retinanet-demo."""

    def __init__(self):
        """Initialize the lightning module."""
        super().__init__()
        self.model = models.detection.retinanet_resnet50_fpn_v2(
            num_classes=len(Config.labels),
            weights_backbone=models.ResNet50_Weights.DEFAULT,
        )

    def train_dataloader(self):
        """Return the train dataloader."""
        return torch.utils.data.DataLoader(
            Dataset(),
            batch_size=Config.batch_size,
            collate_fn=lambda x: x,
        )

    def training_step(self, batch, _):
        """Define the training step."""
        images = [item["image"] for item in batch]
        loss = self.model(images, batch)
        total_loss = loss["classification"] + loss["bbox_regression"]
        kwargs = {"on_step": True, "prog_bar": True, "logger": True}
        self.log("classification_loss", loss["classification"], **kwargs)
        self.log("regression_loss", loss["bbox_regression"], **kwargs)
        return total_loss

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(self.parameters(), lr=Config.learning_rate)


def train_model() -> None:
    """Train the model."""
    model = Lightning()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)


def save_checkpoint(checkpoint_path: Union[str, Path]) -> None:
    """Save the checkpoint."""
    model = Lightning.load_from_checkpoint(checkpoint_path)
    torch.save(model.model.state_dict(), Config.model_path)
    print(f"Save model: dvc add {Config.model_path}")
