from pathlib import Path
import shutil

import pandas as pd


def copy_ultralytics(df, split, out_dir):
    image_dir = Path(out_dir) / "images" / split
    label_dir = Path(out_dir) / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)


def copy_darknet(df, split, out_dir):
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    copied_images = []

    for _, row in df.iterrows():
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])

        new_image = split_dir / image_path.name
        new_label = split_dir / label_path.name

        shutil.copy2(image_path, new_image)
        shutil.copy2(label_path, new_label)

        copied_images.append(new_image)

    return copied_images


def write_image_list(image_paths, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [str(path).replace("\\", "/") for path in image_paths]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_yolo_data(splits_dir, yolo_dir, darknet_dir, yolov4_dir):
    rows = []

    for split in ["train", "val", "test"]:
        df = pd.read_csv(Path(splits_dir) / f"{split}.csv")

        copy_ultralytics(df, split, yolo_dir)
        copied_images = copy_darknet(df, split, darknet_dir)

        list_name = "valid.txt" if split == "val" else f"{split}.txt"
        write_image_list(copied_images, Path(yolov4_dir) / list_name)

        rows.append({"split": split, "n_images": len(df)})

    return pd.DataFrame(rows)
