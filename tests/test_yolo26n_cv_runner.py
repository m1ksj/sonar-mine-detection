from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from train_yolo26n_cv import make_grid, make_run_name  # noqa: E402


def test_make_grid_builds_all_combinations():
    search_space = {
        "augmentation_mode": ["no_aug", "mosaic"],
        "lr0": [0.005, 0.01],
        "batch": [16],
    }

    grid = make_grid(search_space)

    assert len(grid) == 4
    assert {
        "augmentation_mode": "no_aug",
        "lr0": 0.005,
        "batch": 16,
    } in grid
    assert {
        "augmentation_mode": "mosaic",
        "lr0": 0.01,
        "batch": 16,
    } in grid


def test_make_run_name_contains_fold_and_hyperparameters():
    params = {
        "augmentation_mode": "mosaic_affine_color_mild",
        "lr0": 0.005,
        "batch": 16,
        "patience": 50,
        "optimizer": "SGD",
    }

    name = make_run_name(2, params)

    assert name == (
        "fold2_aug-mosaic_affine_color_mild_"
        "lr-0p005_b16_pat50_opt-SGD"
    )
