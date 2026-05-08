import unittest
from pathlib import Path


class TestRepositorySetup(unittest.TestCase):
    def test_expected_top_level_folders_exist(self):
        expected_folders = [
            "configs",
            "data",
            "models",
            "notebooks",
            "reports",
            "scripts",
            "sonar_mine_detection",
            "tests",
        ]

        for folder in expected_folders:
            message = f"Missing folder: {folder}"
            self.assertTrue(Path(folder).exists(), message)

    def test_project_package_exists(self):
        package_file = Path("sonar_mine_detection/__init__.py")
        self.assertTrue(package_file.exists())

    def test_expected_subpackages_exist(self):
        expected_files = [
            "sonar_mine_detection/data/__init__.py",
            "sonar_mine_detection/models/__init__.py",
            "sonar_mine_detection/evaluation/__init__.py",
            "sonar_mine_detection/deployment/__init__.py",
            "sonar_mine_detection/utils/__init__.py",
        ]

        for file_path in expected_files:
            message = f"Missing file: {file_path}"
            self.assertTrue(Path(file_path).exists(), message)


if __name__ == "__main__":
    unittest.main()
