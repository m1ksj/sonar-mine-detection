from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sonar_mine_detection.data.audit import audit_dataset  # noqa: E402


def main():
    with open("configs/paths.yaml", "r", encoding="utf-8") as file:
        paths = yaml.safe_load(file)

    manifest, boxes, issues, summary = audit_dataset(
        raw_dir=paths["raw_data_dir"],
        years=paths["raw_years"],
        metadata_dir=paths["metadata_dir"],
        results_dir=paths["results_dir"],
    )

    print("Audit complete.")
    print(f"Images: {len(manifest)}")
    print(f"Boxes: {len(boxes)}")
    print(f"Issues: {len(issues)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
