"""Train the model."""
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import models

from .config import Config
from .dataset import Dataset
from .utils import Holdout


class Lightning(pl.LightningModule):
    """Lightning module for retinanet-demo."""

    def __init__(self):
        """Initialize the lightning module."""
        super().__init__()
        self.model = models.detection.retinanet_resnet50_fpn_v2(
            num_classes=len(Config.labels),
            weights_backbone=models.ResNet50_Weights.DEFAULT,
        )
        self.dev_map = MeanAveragePrecision()
        self.test_map = MeanAveragePrecision(class_metrics=True)

    def train_dataloader(self):
        """Return the train dataloader."""
        return torch.utils.data.DataLoader(
            Dataset(Holdout.train),
            batch_size=Config.batch_size,
            collate_fn=lambda x: x,
            worker_init_fn=lambda _: np.random.seed(),
            shuffle=True,
            num_workers=Config.num_workers,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return torch.utils.data.DataLoader(
            Dataset(Holdout.dev),
            batch_size=Config.batch_size,
            collate_fn=lambda x: x,
            num_workers=Config.num_workers,
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        return torch.utils.data.DataLoader(
            Dataset(Holdout.test),
            batch_size=Config.batch_size,
            collate_fn=lambda x: x,
            num_workers=Config.num_workers,
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

    def validation_step(self, batch, _):
        """Define the validation step."""
        images = [item["image"] for item in batch]
        predictions = self.model(images)
        self.dev_map.update(predictions, batch)

    def test_step(self, batch, _):
        """Define the test step."""
        images = [item["image"] for item in batch]
        predictions = self.model(images)
        self.test_map.update(predictions, batch)

    def on_validation_epoch_end(self) -> None:
        """Define metrics for each validation epoch."""
        map_scores = self.dev_map.compute()
        self.log("dev_map", map_scores["map"], on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> dict[str, torch.Tensor]:
        """Define metrics for each test epoch."""
        metrics = self.test_map.compute()
        log_metrics = OrderedDict()
        for k, v in metrics.items():
            if len(v.shape) == 0:
                log_metrics[k] = v
            else:
                for i, score in enumerate(v):
                    class_label = Config.labels[i]
                    log_metrics[f"map_{class_label}"] = score
        self.log_dict(log_metrics)

    def configure_optimizers(self):
        """Configure the optimizers."""
        return torch.optim.Adam(self.parameters(), lr=Config.learning_rate)


def train_model() -> None:
    """Train the model."""
    model = Lightning()
    trainer = pl.Trainer(
        max_epochs=6,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping("dev_map", mode="max")],
    )
    trainer.fit(model)


def save_checkpoint(checkpoint_path: Union[str, Path]) -> None:
    """Save the checkpoint."""
    model = Lightning.load_from_checkpoint(checkpoint_path)
    torch.save(model.model.state_dict(), Config.model_path)
    print(f"Save model: dvc add {Config.model_path}")
