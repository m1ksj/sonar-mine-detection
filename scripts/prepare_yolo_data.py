from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    from sonar_mine_detection.data.yolo_dataset import prepare_yolo_data

    with open("configs/paths.yaml", "r", encoding="utf-8") as file:
        paths = yaml.safe_load(file)

    summary = prepare_yolo_data(
        splits_dir=paths["splits_dir"],
        yolo_dir=paths["processed_yolo_dir"],
        darknet_dir=paths["processed_darknet_dir"],
        yolov4_dir="configs/yolov4",
    )

    output = Path(paths["results_dir"]) / "yolo_data_export_summary.csv"
    summary.to_csv(output, index=False)

    print("YOLO data export complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
