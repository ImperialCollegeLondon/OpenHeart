"""Build nnU-Net v2 training datasets from labelled cases."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from .image import Image
from .labels import LabelMap

log = logging.getLogger(__name__)

LABELS = {"background": 0, "left_ventricle": 1, "myocardium": 2, "right_ventricle": 3}


def pairs_from_csv(csv_path, base_dir: str = "") -> list:
    """(image, label) path pairs from a CSV with ``image_path,label_path`` columns."""
    with open(csv_path, newline="") as f:
        return [(base_dir + row["image_path"], base_dir + row["label_path"]) for row in csv.DictReader(f)]


def export_nnunet_dataset(items, nnunet_raw, dataset_id: int, name: str, labels=None,
                          description: str = "OpenHeart cardiac MRI segmentation") -> Path:
    """Write ``Dataset<id>_<name>/{imagesTr,labelsTr,dataset.json}`` under ``nnunet_raw``.

    ``items`` yields either ``(image_path, label_path)`` pairs or Cases with
    segmentations (one training sample per phase). ``labels`` is the label
    convention of the files (paths only; Cases are already canonical). Pairs whose
    image and label grids differ are skipped.
    """
    labels = LabelMap.from_spec(labels)
    root = Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{name}"
    images_dir, labels_dir = root / "imagesTr", root / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    def samples():
        for item in items:
            if isinstance(item, (tuple, list)):
                image, seg = Image.load(item[0]), Image.load(item[1])
                yield str(item[0]), image, seg.with_array(labels.to_canonical(np.rint(seg.array)))
            else:
                for phase, seg in item.segs.items():
                    if phase in item.images:
                        yield f"{item.id}/{phase}", item.images[phase], seg

    n, skipped = 0, 0
    for source, image, seg in samples():
        if image.shape != seg.shape:
            log.warning("skipping %s: image %s vs label %s", source, image.shape, seg.shape)
            skipped += 1
            continue
        image.save(images_dir / f"case_{n:04d}_0000.nii.gz")
        seg.with_array(seg.array.astype(np.uint8)).save(labels_dir / f"case_{n:04d}.nii.gz")
        n += 1
    (root / "dataset.json").write_text(json.dumps({
        "name": name, "description": description, "channel_names": {"0": "MRI"},
        "labels": LABELS, "numTraining": n, "file_ending": ".nii.gz",
    }, indent=2))
    log.info("wrote %d training cases to %s (%d skipped)", n, root, skipped)
    return root
