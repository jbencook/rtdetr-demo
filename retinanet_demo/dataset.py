"""Prepare the dataset for training and testing."""
from pathlib import Path
from tempfile import TemporaryDirectory

import datumaro as dm
import torch
from PIL import Image
from sparrow_cvat import download_annotations, download_images
from sparrow_datums import FrameAugmentedBoxes
from torchvision.transforms import ToTensor

from .config import Config
from .utils import Holdout, in_holdout


def version_images(task_id: int) -> None:
    """Download the CVAT images."""
    download_images(task_id, Config.images_directory)
    print(f"Version images: dvc add {Config.images_directory}")


def version_annotations(task_id: int) -> None:
    """Version the CVAT annotations."""
    with TemporaryDirectory() as tmpdir:
        cvat_path = Path(tmpdir) / "annotations.xml"
        download_annotations(task_id, cvat_path)
        dataset = dm.Dataset.import_from(cvat_path, "cvat")
    for dataset_item in dataset:
        boxes = FrameAugmentedBoxes.from_dataset_item(dataset_item)
        boxes.to_file(Config.annotations_directory / f"{dataset_item.id}.json.gz")
    print(f"Version annotations: dvc add {Config.annotations_directory}")


class Dataset(torch.utils.data.Dataset):
    """Dataset class for retinanet-demo."""

    def __init__(self, holdout: Holdout) -> None:
        """Initialize the dataset."""
        self.slugs = []
        for image_path in Config.images_directory.glob("*.jpg"):
            slug = image_path.stem
            if in_holdout(slug, holdout):
                self.slugs.append(slug)
        self.transform = ToTensor()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.slugs)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return an item from the dataset."""
        slug = self.slugs[index]
        image = Image.open(Config.images_directory / f"{slug}.jpg")
        boxes = FrameAugmentedBoxes.from_file(
            Config.annotations_directory / f"{slug}.json.gz"
        )
        boxes = boxes.to_absolute().to_tlbr()
        return {
            "image": self.transform(image),
            "boxes": torch.from_numpy(boxes.array[..., :4].astype("float32")),
            "labels": torch.from_numpy(boxes.labels.astype("int32")),
        }
