"""Prepare the dataset for training and testing."""
from pathlib import Path
from tempfile import TemporaryDirectory

import datumaro as dm
from sparrow_cvat import download_annotations
from sparrow_datums import FrameAugmentedBoxes


def version_annotations(task_id: int) -> None:
    """Version the CVAT annotations."""
    with TemporaryDirectory() as tmpdir:
        cvat_path = Path(tmpdir) / "annotations.xml"
        download_annotations(task_id, cvat_path)
        dataset = dm.Dataset.import_from(cvat_path, "cvat")
    for dataset_item in dataset:
        boxes = FrameAugmentedBoxes.from_dataset_item(dataset_item)
        print(dataset_item, boxes)
