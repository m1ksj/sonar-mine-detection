# YOLOv4 Darknet baseline

YOLOv4 is used as an external Darknet baseline, not through Ultralytics.

- keep Darknet outside this repository!
- run YOLOv4 later in Google Colab, Linux where Darknet can be compiled.
- commit only our configuration notes, class names and result summaries.

## Classes

0 = MILCO
1 = NOMBO

## Expected Darknet data structure

For Darknet, each image should have its matching label file next to it:

data/obj/train/
├── image_001.jpg
├── image_001.txt
└── ...

data/obj/val/
├── image_101.jpg
├── image_101.txt
└── ...


## Training command

./darknet detector train data/obj/obj.data cfg/yolov4-sonar.cfg yolov4.conv.137 -dont_show -map

## Evaluation command

./darknet detector map data/obj/obj.data cfg/yolov4-sonar.cfg backup/yolov4-sonar_best.weights -iou_thresh 0.50
