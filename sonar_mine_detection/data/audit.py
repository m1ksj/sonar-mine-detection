from pathlib import Path

import pandas as pd
from PIL import Image

CLASS_NAMES = {0: "MILCO", 1: "NOMBO"}


def read_labels(label_path):
    labels = []
    issues = []

    if not label_path.exists():
        return labels, ["missing_label_file"]

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue

        parts = line.split()

        if len(parts) != 5:
            issues.append("wrong_label_format")
            continue

        class_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        if class_id not in [0, 1]:
            issues.append("invalid_class_id")

        valid_values = 0 <= x <= 1 and 0 <= y <= 1
        valid_size = 0 < w <= 1 and 0 < h <= 1

        if not (valid_values and valid_size):
            issues.append("invalid_bbox_values")

        if x - w / 2 < 0 or x + w / 2 > 1:
            issues.append("bbox_outside_image")

        if y - h / 2 < 0 or y + h / 2 > 1:
            issues.append("bbox_outside_image")

        labels.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, "invalid"),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    return labels, issues


def get_category(class_ids):
    if 0 in class_ids and 1 in class_ids:
        return "mixed"
    if 0 in class_ids:
        return "milco_only"
    if 1 in class_ids:
        return "nombo_only"
    return "empty"


def audit_dataset(raw_dir, years, metadata_dir, results_dir):
    raw_dir = Path(raw_dir)
    metadata_dir = Path(metadata_dir)
    results_dir = Path(results_dir)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    bbox_rows = []
    issue_rows = []

    for year in years:
        year_dir = raw_dir / str(year) / str(year)

        for image_path in sorted(year_dir.glob("*.jpg")):
            label_path = image_path.with_suffix(".txt")

            with Image.open(image_path) as image:
                width, height = image.size

            labels, issues = read_labels(label_path)
            class_ids = [label["class_id"] for label in labels]

            manifest_rows.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "year": year,
                    "width": width,
                    "height": height,
                    "resolution": f"{width}x{height}",
                    "n_objects": len(labels),
                    "has_milco": 0 in class_ids,
                    "has_nombo": 1 in class_ids,
                    "image_category": get_category(class_ids),
                }
            )

            for label in labels:
                bbox_rows.append(
                    {
                        "image_id": image_path.stem,
                        "year": year,
                        "class_id": label["class_id"],
                        "class_name": label["class_name"],
                        "box_width_px": label["w"] * width,
                        "box_height_px": label["h"] * height,
                    }
                )

            for issue in sorted(set(issues)):
                issue_rows.append(
                    {
                        "image_id": image_path.stem,
                        "year": year,
                        "issue": issue,
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    boxes = pd.DataFrame(bbox_rows)
    issues = pd.DataFrame(issue_rows)

    summary = pd.DataFrame(
        [
            {
                "n_images": len(manifest),
                "n_boxes": len(boxes),
                "n_issues": len(issues),
                "n_empty": (manifest["image_category"] == "empty").sum(),
                "n_milco_only": (
                    manifest["image_category"] == "milco_only"
                ).sum(),
                "n_nombo_only": (
                    manifest["image_category"] == "nombo_only"
                ).sum(),
                "n_mixed": (manifest["image_category"] == "mixed").sum(),
            }
        ]
    )

    manifest.to_csv(metadata_dir / "dataset_manifest.csv", index=False)
    boxes.to_csv(results_dir / "bbox_summary.csv", index=False)
    issues.to_csv(results_dir / "invalid_annotations.csv", index=False)
    summary.to_csv(results_dir / "audit_summary.csv", index=False)

    return manifest, boxes, issues, summary
