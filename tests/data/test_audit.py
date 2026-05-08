import tempfile
import unittest
from pathlib import Path

from sonar_mine_detection.data.audit import get_category, read_labels


class TestAudit(unittest.TestCase):

    def test_get_category(self):
        self.assertEqual(get_category([]), "empty")
        self.assertEqual(get_category([0]), "milco_only")
        self.assertEqual(get_category([1]), "nombo_only")
        self.assertEqual(get_category([0, 1]), "mixed")

    def test_read_valid_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = Path(tmpdir) / "image.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            labels, issues = read_labels(label_path)

            self.assertEqual(len(labels), 1)
            self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
