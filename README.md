# Yolov4 tiny face detection

Cfg: yolov4-tiny-3l

Dataset: WIDER FACE

## Original video:

https://www.youtube.com/watch?v=1_6dMpjXWMw

## Video inference on RTX2080Ti (416x416)

https://www.youtube.com/watch?v=uOWMoYjL9HI

## Video inference on Jetson nano (192x192)

Lagging due to reading video time from SD Card on jetson nano

https://www.youtube.com/watch?v=EvWIY7NROpg

## Run

Build darknet for GPU, CUDNN, LIBSO from github repo:

https://github.com/AlexeyAB/darknet

Copy libdarknet.so to project directory


```
python3 face_detection.py
```
