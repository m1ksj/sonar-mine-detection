from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from ultralytics import __version__ as ultralytics_version
from ultralytics.cfg import DEFAULT_CFG_DICT


YOLOV4_CFG_URL = (
    "https://raw.githubusercontent.com/AlexeyAB/darknet/"
    "master/cfg/yolov4-custom.cfg"
)


def read_url(url):
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def parse_darknet_cfg(text):
    rows = []
    section = None
    section_index = -1

    for line_number, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()

        if not raw or raw.startswith("#"):
            continue

        if raw.startswith("[") and raw.endswith("]"):
            section = raw.strip("[]")
            section_index += 1
            continue

        if "=" not in raw:
            continue

        key, value = raw.split("=", 1)

        rows.append(
            {
                "source": YOLOV4_CFG_URL,
                "section": section,
                "section_index": section_index,
                "line_number": line_number,
                "parameter": key.strip(),
                "value": value.strip(),
            }
        )

    return pd.DataFrame(rows)


def main():
    output_dir = Path("reports/results_tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    yolov4_text = read_url(YOLOV4_CFG_URL)
    yolov4_df = parse_darknet_cfg(yolov4_text)
    yolov4_output = output_dir / "yolov4_default_cfg_audit.csv"
    yolov4_df.to_csv(yolov4_output, index=False)

    yolo26n_df = pd.DataFrame(
        [
            {
                "source": (
                    "Ultralytics DEFAULT_CFG_DICT "
                    f"version {ultralytics_version}"
                ),
                "parameter": key,
                "value": value,
            }
            for key, value in DEFAULT_CFG_DICT.items()
        ]
    )

    yolo26n_output = output_dir / "yolo26n_default_finetune_args_audit.csv"
    yolo26n_df.to_csv(yolo26n_output, index=False)

    print(f"Saved {yolov4_output}")
    print(f"Saved {yolo26n_output}")


if __name__ == "__main__":
    main()
