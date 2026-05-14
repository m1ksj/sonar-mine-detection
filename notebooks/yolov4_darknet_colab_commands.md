# YOLOv4 Darknet Colab commands

This file documents how to reproduce the YOLOv4 baseline in Google Colab.

YOLOv4 is trained once as a baseline. We do not run 5-fold CV for YOLOv4.

## 1. Clone the project repository

    !git clone https://github.com/m1ksj/sonar-mine-detection.git
    %cd sonar-mine-detection

## 2. Prepare the dataset

Place the raw dataset under:

    data/raw/2010/2010/
    data/raw/2015/2015/
    data/raw/2017/2017/
    data/raw/2018/2018/
    data/raw/2021/2021/

Then run:

    !pip install pipenv
    !pipenv install
    !pipenv run python scripts/audit_dataset.py
    !pipenv run python scripts/create_splits.py
    !pipenv run python scripts/prepare_yolo_data.py

## 3. Clone and build Darknet

    %cd /content
    !git clone https://github.com/AlexeyAB/darknet
    %cd darknet

    !sed -i 's/GPU=0/GPU=1/' Makefile
    !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    !sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
    !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    !make

## 4. Download pretrained weights

    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

## 5. Create custom YOLOv4 config

    !cp cfg/yolov4-custom.cfg cfg/yolov4-sonar.cfg

Edit cfg/yolov4-sonar.cfg:

    width=640
    height=640
    batch=64
    subdivisions=16
    max_batches=6000
    steps=4800,5400
    classes=2 in all 3 YOLO layers
    filters=21 before all 3 YOLO layers

## 6. Train YOLOv4

From the darknet folder:

    !./darknet detector train /content/sonar-mine-detection/configs/yolov4/obj.data cfg/yolov4-sonar.cfg yolov4.conv.137 -dont_show -map

## 7. Evaluate YOLOv4

    !./darknet detector map /content/sonar-mine-detection/configs/yolov4/obj.data cfg/yolov4-sonar.cfg backup/yolov4-sonar_best.weights -iou_thresh 0.50

## 8. Save results

Save the best weights and metrics externally, for example in Google Drive.

Do not commit Darknet, raw data, processed data, or checkpoints to GitHub.
