"""WatermarkSubtitlesDatasetSegmentation dataset"""

import os
import glob

import numpy as np
import cv2
import tensorflow_datasets as tfds

_DESCRIPTION = """
Dataset for summer practice coursework
on watermark and subtitles removal
project.
"""

_CITATION = """
"""


class WatermarkSubtitlesDatasetSegmentation(tfds.core.GeneratorBasedBuilder):
    """Dataset for segmentation of subtitles and watermarks"""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(
                        shape=(None, None, 3),
                        dtype=np.uint8,
                    ),
                    "target_mask": tfds.features.Image(
                        shape=(None, None, 1),
                        dtype=np.uint8,
                    ),
                }
            ),
            supervised_keys=("image", "target_mask"),
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        data_dir = "data/generated"

        return {
            "train": self._generate_examples(f"{data_dir}/train", count=1000),
            "validate": self._generate_examples(f"{data_dir}/validate", count=500),
        }

    def _generate_examples(self, folder: str, count: int = None):
        """Yields examples."""
        for img_idx, img_path in enumerate(glob.glob(f"{folder}/*.jpg")):
            basename = os.path.basename(img_path)

            image = cv2.imread(img_path).astype(np.uint8)
            segmentation_mask = f"{folder}/{basename.replace('.jpg', '.npz')}"
            segmentation_mask = np.load(segmentation_mask)["mask"].astype(np.uint8)
            segmentation_mask = np.expand_dims(segmentation_mask, axis=-1)

            yield basename, {"image": image, "target_mask": segmentation_mask}
            if count is not None and img_idx > count:
                break
