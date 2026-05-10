# YOLOv4 Darknet baseline

YOLOv4 is used as an external Darknet baseline, not through Ultralytics.

We use YOLOv4 as a historical reference because the dataset paper already used YOLOv4 on this exact side-scan sonar dataset.

## Decision

YOLOv4 will be trained once as a full baseline run.

We do not run 5-fold CV for YOLOv4. Darknet training is more expensive and less integrated into our Python pipeline. Wider tuning and cross-validation are reserved for YOLO26n, which is the main model.

## Classes

0 = MILCO
1 = NOMBO

## Data files

The data export script creates the local Darknet-style data folders:

    data/processed/darknet/train/
    data/processed/darknet/val/
    data/processed/darknet/test/

The image list files are:

    configs/yolov4/train.txt
    configs/yolov4/valid.txt
    configs/yolov4/test.txt

The Darknet data config is:

    configs/yolov4/obj.data

## YOLOv4 settings

The original dataset paper trained YOLOv4 with:

    batch=64
    subdivisions=16
    max_batches=6000
    steps=4800,5400
    pretrained weights=yolov4.conv.137
    input size=512x512

For our project, we keep the same main YOLOv4 training settings but use:

    width=640
    height=640

This keeps YOLOv4 comparable to YOLO26n.

## Required config edits

Use Darknet's yolov4-custom.cfg as base and save the edited file as:

    cfg/yolov4-sonar.cfg

Required changes:

    width=640
    height=640
    batch=64
    subdivisions=16
    max_batches=6000
    steps=4800,5400

Since this project has 2 classes, set this in all 3 YOLO layers:

    classes=2

In the convolutional layer before each YOLO layer, set:

    filters=21

Formula:

    filters = (classes + 5) * 3
    filters = (2 + 5) * 3 = 21

If training runs out of memory, increase subdivisions to 32 or 64.

## Training command

    ./darknet detector train configs/yolov4/obj.data cfg/yolov4-sonar.cfg yolov4.conv.137 -dont_show -map

## Evaluation command

    ./darknet detector map configs/yolov4/obj.data cfg/yolov4-sonar.cfg models/checkpoints/yolov4/yolov4-sonar_best.weights -iou_thresh 0.50

## Do not commit

Do not commit:

    darknet/
    data/processed/
    models/checkpoints/
    *.weights
