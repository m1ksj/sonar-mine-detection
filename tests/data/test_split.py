import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sonar_mine_detection.data.split import create_splits


class TestSplit(unittest.TestCase):

    def test_split_has_no_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            rows = []
            categories = ["empty", "milco_only", "nombo_only", "mixed"]

            for i in range(80):
                rows.append(
                    {
                        "image_id": f"img_{i}",
                        "image_category": categories[i % 4],
                        "year": 2018,
                        "resolution": "416x416",
                    }
                )

            manifest_path = tmpdir / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)

            train, val, test = create_splits(
                manifest_path=manifest_path,
                splits_dir=tmpdir / "splits",
                results_dir=tmpdir / "results",
                seed=42,
                n_folds=5,
            )

            train_ids = set(train["image_id"])
            val_ids = set(val["image_id"])
            test_ids = set(test["image_id"])

            self.assertFalse(train_ids & val_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(val_ids & test_ids)

            fold_val = pd.read_csv(tmpdir / "splits" / "fold_0_val.csv")
            fold_val_ids = set(fold_val["image_id"])

            self.assertTrue(fold_val_ids.issubset(train_ids))


if __name__ == "__main__":
    unittest.main()
