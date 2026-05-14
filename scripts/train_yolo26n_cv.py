from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

import pandas as pd
import yaml


CLASS_NAMES = {0: "MILCO", 1: "NOMBO"}


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def make_grid(search_space: dict) -> list[dict]:
    keys = list(search_space)
    values = [search_space[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def safe_name(value: object) -> str:
    return str(value).replace(".", "p").replace("/", "-")


def make_run_name(fold: int, params: dict) -> str:
    aug = safe_name(params["augmentation_mode"])
    lr0 = safe_name(params["lr0"])
    batch = safe_name(params["batch"])
    patience = safe_name(params["patience"])
    optimizer = safe_name(params["optimizer"])
    return (
        f"fold{fold}_aug-{aug}_lr-{lr0}_b{batch}_"
        f"pat{patience}_opt-{optimizer}"
    )


def build_file_index(data_root: Path) -> dict[str, Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".txt"}
    return {
        path.name: path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def find_file(path_from_csv: str, file_index: dict[str, Path]) -> Path:
    path = Path(path_from_csv.replace("\\", "/"))

    if path.exists():
        return path

    if path.name in file_index:
        return file_index[path.name]

    raise FileNotFoundError(f"Missing data file: {path_from_csv}")


def copy_split(
    csv_path: Path,
    split_name: str,
    dataset_dir: Path,
    file_index: dict[str, Path],
) -> int:
    df = pd.read_csv(csv_path)

    image_dir = dataset_dir / "images" / split_name
    label_dir = dataset_dir / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        image_path = find_file(row["image_path"], file_index)
        label_path = find_file(row["label_path"], file_index)

        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)

    return len(df)


def write_data_yaml(dataset_dir: Path, include_test: bool = False) -> Path:
    data_yaml = dataset_dir / "data.yaml"
    data = {
        "path": str(dataset_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }

    if include_test:
        data["test"] = "images/test"

    save_yaml(data, data_yaml)
    return data_yaml


def prepare_fold_dataset(
    split_dir: Path,
    fold: int,
    work_dir: Path,
    file_index: dict[str, Path],
) -> Path:
    dataset_dir = work_dir / f"fold_{fold}"

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    copy_split(split_dir / f"fold_{fold}_train.csv",
               "train", dataset_dir, file_index)
    copy_split(split_dir / f"fold_{fold}_val.csv",
               "val", dataset_dir, file_index)

    return write_data_yaml(dataset_dir)


def prepare_final_dataset(
    split_dir: Path,
    work_dir: Path,
    file_index: dict[str, Path],
) -> Path:
    dataset_dir = work_dir / "final_train_val_test"

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    copy_split(split_dir / "train.csv", "train", dataset_dir, file_index)
    copy_split(split_dir / "val.csv", "val", dataset_dir, file_index)
    copy_split(split_dir / "test.csv", "test", dataset_dir, file_index)

    return write_data_yaml(dataset_dir, include_test=True)


def find_metric_column(columns: list[str], wanted: str) -> str | None:
    wanted = wanted.lower()

    for column in columns:
        clean = column.strip().lower()

        if wanted == "map50_95" and "map50-95" in clean:
            return column

        if wanted == "map50" and "map50" in clean and "95" not in clean:
            return column

        if wanted == "precision" and "precision" in clean:
            return column

        if wanted == "recall" and "recall" in clean:
            return column

    return None


def read_run_metrics(run_dir: Path) -> dict:
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        return {"status": "missing_results"}

    results = pd.read_csv(results_csv)
    results.columns = [column.strip() for column in results.columns]
    columns = list(results.columns)

    map50_95_col = find_metric_column(columns, "map50_95")
    map50_col = find_metric_column(columns, "map50")
    precision_col = find_metric_column(columns, "precision")
    recall_col = find_metric_column(columns, "recall")

    if map50_95_col is not None:
        best_idx = pd.to_numeric(
            results[map50_95_col], errors="coerce").idxmax()
    elif map50_col is not None:
        best_idx = pd.to_numeric(results[map50_col], errors="coerce").idxmax()
    else:
        best_idx = len(results) - 1

    best = results.loc[best_idx]
    row = {"status": "ok", "best_epoch": int(best_idx)}

    if map50_95_col is not None:
        row["map50_95"] = float(best[map50_95_col])
    if map50_col is not None:
        row["map50"] = float(best[map50_col])
    if precision_col is not None:
        row["precision"] = float(best[precision_col])
    if recall_col is not None:
        row["recall"] = float(best[recall_col])

    return row


def save_tables(rows: list[dict], output_dir: Path) -> pd.DataFrame:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)
    results.to_csv(table_dir / "yolo26n_cv_results.csv", index=False)

    if results.empty:
        return results

    group_cols = ["augmentation_mode", "lr0", "batch", "patience", "optimizer"]
    metric_cols = [
        col
        for col in ["map50_95", "map50", "precision", "recall"]
        if col in results
    ]

    summary = (
        results.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    for metric in metric_cols:
        summary[f"{metric}_se"] = summary[f"{metric}_std"] / summary[
            f"{metric}_count"
        ].pow(0.5)

    summary.to_csv(table_dir / "yolo26n_cv_summary.csv", index=False)
    return results


def train_yolo(
    model_name: str,
    data_yaml: Path,
    output_dir: Path,
    run_name: str,
    train_params: dict,
    aug_params: dict,
    device: str | None,
    workers: int,
) -> Path:
    from ultralytics import YOLO

    project_dir = output_dir / "ultralytics_runs"

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        imgsz=int(train_params["imgsz"]),
        epochs=int(train_params["epochs"]),
        batch=int(train_params["batch"]),
        optimizer=str(train_params["optimizer"]),
        lr0=float(train_params["lr0"]),
        patience=int(train_params["patience"]),
        seed=int(train_params["seed"]),
        workers=int(workers),
        device=device,
        plots=bool(train_params.get("plots", True)),
        cache=bool(train_params.get("cache", False)),
        save=True,
        verbose=True,
        **aug_params,
    )

    return project_dir / run_name


def select_best_config(results: pd.DataFrame) -> dict:
    group_cols = ["augmentation_mode", "lr0", "batch", "patience", "optimizer"]
    metric = "map50_95" if "map50_95" in results.columns else "map50"

    grouped = (
        results.groupby(group_cols, dropna=False)[metric]
        .mean()
        .reset_index()
        .sort_values(metric, ascending=False)
    )

    return grouped.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yolo26n_tuning.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", default="yolo26n_cv_work")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train-final", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    aug_config = load_yaml(config["augmentation_config"])
    runtime = config.get("runtime", {})

    split_dir = Path(config["cv"]["split_dir"])
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    data_root = Path(args.data_root)

    grid = make_grid(config["search_space"])
    folds = list(config["cv"]["folds"])

    if args.debug:
        grid = grid[:1]
        folds = folds[:1]
        config["epochs"] = 2

    for params in grid:
        mode = params["augmentation_mode"]
        if mode not in aug_config["modes"]:
            raise ValueError(f"Unknown augmentation mode: {mode}")

    n_runs = len(grid) * len(folds)
    print(f"Planned CV runs: {n_runs}")
    print(f"Folds: {folds}")

    if args.dry_run:
        print("Dry run complete. No training started.")
        return

    file_index = build_file_index(data_root)
    if not file_index:
        raise FileNotFoundError(
            f"No image/label files found under {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(config, output_dir / "used_yolo26n_tuning.yaml")
    save_yaml(aug_config, output_dir / "used_augmentation_yolo26n.yaml")

    rows = []

    for fold in folds:
        data_yaml = prepare_fold_dataset(split_dir, fold, work_dir, file_index)

        for params in grid:
            aug_params = aug_config["modes"][params["augmentation_mode"]]
            run_name = make_run_name(fold, params)

            train_params = {
                "imgsz": config["imgsz"],
                "epochs": config["epochs"],
                "seed": config["seed"],
                "plots": runtime.get("plots_cv", False),
                "cache": runtime.get("cache", True),
                **params,
            }

            print(f"\nStarting {run_name}")

            run_dir = train_yolo(
                model_name=config["model"],
                data_yaml=data_yaml,
                output_dir=output_dir,
                run_name=run_name,
                train_params=train_params,
                aug_params=aug_params,
                device=args.device,
                workers=args.workers,
            )

            row = {"fold": fold, **train_params, **read_run_metrics(run_dir)}
            rows.append(row)
            results = save_tables(rows, output_dir)

    results = save_tables(rows, output_dir)

    if args.train_final:
        best = select_best_config(results)
        best_mode = best["augmentation_mode"]

        final_config = config.get("final", {})
        final_patience = final_config.get(
                "patience",
                int(best["patience"]),
        )

        final_params = {
                "imgsz": config["imgsz"],
                "epochs": int(final_config.get("epochs", config["epochs"])),
                "seed": config["seed"],
                "augmentation_mode": best_mode,
                "lr0": float(best["lr0"]),
                "batch": int(best["batch"]),
                "patience": int(final_patience),
                "optimizer": best["optimizer"],
                "plots": runtime.get("plots_final", True),
                "cache": runtime.get("cache", True),
        }

        final_yaml = prepare_final_dataset(split_dir, work_dir, file_index)

        save_yaml(
            {
                "selected_config": final_params,
                "augmentation": aug_config["modes"][best_mode],
            },
            output_dir / "final_selected_config.yaml",
        )

        final_dir = train_yolo(
            model_name=config["model"],
            data_yaml=final_yaml,
            output_dir=output_dir,
            run_name="final_yolo26n_best_cv_config",
            train_params=final_params,
            aug_params=aug_config["modes"][best_mode],
            device=args.device,
            workers=args.workers,
        )

        print(f"Final model saved in: {final_dir}")
        final_weights = final_dir / "weights" / "best.pt"

        if final_weights.exists():
            from ultralytics import YOLO

            model = YOLO(str(final_weights))
            test_metrics = model.val(
                data=str(final_yaml),
                split="test",
                project=str(output_dir / "ultralytics_runs"),
                name="final_yolo26n_test_eval",
                imgsz=int(final_params["imgsz"]),
                batch=int(final_params["batch"]),
                workers=int(args.workers),
                device=args.device,
                plots=True,
            )

            test_row = {
                "precision": float(test_metrics.box.mp),
                "recall": float(test_metrics.box.mr),
                "map50": float(test_metrics.box.map50),
                "map50_95": float(test_metrics.box.map),
            }

            table_dir = output_dir / "tables"
            table_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([test_row]).to_csv(
                table_dir / "final_yolo26n_test_metrics.csv",
                index=False,
            )

            print("Held-out test metrics:")
            print(test_row)

    print("Done.")


if __name__ == "__main__":
    main()
