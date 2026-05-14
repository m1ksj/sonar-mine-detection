from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    from sonar_mine_detection.data.split import create_splits

    with open("configs/paths.yaml", "r", encoding="utf-8") as file:
        paths = yaml.safe_load(file)

    with open("configs/split.yaml", "r", encoding="utf-8") as file:
        split_config = yaml.safe_load(file)

    train, val, test = create_splits(
        manifest_path=f"{paths['metadata_dir']}/dataset_manifest.csv",
        splits_dir=paths["splits_dir"],
        results_dir=paths["results_dir"],
        seed=split_config["seed"],
        n_folds=split_config["n_folds"],
    )

    print("Splits created.")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")


if __name__ == "__main__":
    main()
