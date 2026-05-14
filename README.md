# Real-World Side-Scan Sonar Mine Detection with YOLO26n

## Goal

This project investigates object detection for MILCO/NOMBO detection in real-world side-scan sonar images.

The planned main model is YOLO26n through Ultralytics. YOLOv4 is planned as a competitive historical baseline through Darknet.


## Setup

Install dependencies:

```powershell
py -m pipenv install
```

Activate the environment:

```powershell
py -m pipenv shell
```

Install pre-commit hooks:

```powershell
pre-commit install
```

Run tests:

```powershell
python -m unittest discover -v tests
```

Run pre-commit manually:

```powershell
pre-commit run --all-files
```

## Data

Raw data is not committed.

The downloaded dataset should be placed locally under:

```text
data/raw/
├── 2010/
│   └── 2010/
│       ├── image_001.jpg
│       ├── image_001.txt
│       └── ...
├── 2015/
│   └── 2015/
├── 2017/
│   └── 2017/
├── 2018/
│   └── 2018/
└── 2021/
    └── 2021/
```


YOLO label format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Class IDs:

```text
0 = MILCO
1 = NOMBO
```

## Deployment

Deployment will be added later after a trained model exists

The planned structure is:

```text
sonar_mine_detection/deployment/
├── inference.py
└── api.py
```

The API will later be served with FastAPI and Uvicorn.

## Git rules

- Work on personal branches, not directly on `main`
- Prefer merging finished work into `dev`
- Do not commit raw data, processed data, training runs or checkpoints