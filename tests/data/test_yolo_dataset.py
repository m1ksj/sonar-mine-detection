import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from sonar_mine_detection.data.yolo_dataset import prepare_yolo_data


class TestYoloDataset(unittest.TestCase):

    def test_prepare_yolo_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_dir = tmpdir / "raw"
            splits_dir = tmpdir / "splits"

            raw_dir.mkdir()
            splits_dir.mkdir()

            image_path = raw_dir / "image.jpg"
            label_path = raw_dir / "image.txt"

            Image.new("RGB", (32, 32)).save(image_path)
            label_path.write_text("0 0.5 0.5 0.2 0.2\n")

            row = {
                "image_path": str(image_path),
                "label_path": str(label_path),
            }

            for split in ["train", "val", "test"]:
                pd.DataFrame([row]).to_csv(
                    splits_dir / f"{split}.csv",
                    index=False,
                )

            summary = prepare_yolo_data(
                splits_dir=splits_dir,
                yolo_dir=tmpdir / "yolo",
                darknet_dir=tmpdir / "darknet",
                yolov4_dir=tmpdir / "yolov4",
            )

            self.assertEqual(len(summary), 3)
            self.assertTrue(
                (tmpdir / "yolo/images/train/image.jpg").exists()
            )
            self.assertTrue(
                (tmpdir / "darknet/train/image.txt").exists()
            )
            self.assertTrue((tmpdir / "yolov4/train.txt").exists())


if __name__ == "__main__":
    unittest.main()
