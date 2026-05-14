from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_summary(train, val, test):
    rows = []

    for name, df in [("train", train), ("val", val), ("test", test)]:
        rows.append(
            {
                "split": name,
                "variable": "total",
                "value": "all",
                "count": len(df),
            }
        )

        for column in ["image_category", "year", "resolution"]:
            counts = df[column].value_counts()

            for value, count in counts.items():
                rows.append(
                    {
                        "split": name,
                        "variable": column,
                        "value": value,
                        "count": count,
                    }
                )

    return pd.DataFrame(rows)


def create_splits(
    manifest_path,
    splits_dir,
    results_dir,
    seed,
    n_folds,
):
    manifest = pd.read_csv(manifest_path)

    train, temp = train_test_split(
        manifest,
        test_size=0.30,
        random_state=seed,
        stratify=manifest["image_category"],
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=seed,
        stratify=temp["image_category"],
    )

    splits_dir = Path(splits_dir)
    results_dir = Path(results_dir)

    save_csv(train, splits_dir / "train.csv")
    save_csv(val, splits_dir / "val.csv")
    save_csv(test, splits_dir / "test.csv")

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train, train["image_category"])
    ):
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]

        save_csv(fold_train, splits_dir / f"fold_{fold}_train.csv")
        save_csv(fold_val, splits_dir / f"fold_{fold}_val.csv")

    summary = make_summary(train, val, test)
    save_csv(summary, results_dir / "split_summary.csv")

    return train, val, test
