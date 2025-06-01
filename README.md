This project performs object detection on a video using the YOLOv8 (Ultralytics) model. Detected objects are drawn in real time using OpenCV. You can also track coordinates with the mouse.

Features:

✅ Object detection via YOLOv8

✅ Frame-by-frame analysis of objectdetection.mp4

✅ Class matching with coco.txt

✅ Real-time visualization with OpenCV

✅ Box data analysis with Pandas

Requirements:

pip install opencv-python ultralytics pandas numpy

Usage:

Create a coco.txt file with class labels.

Ensure objectdetection.mp4 is in the same directory.

Download and place the yolov8s.pt model in the project folder.

Run the script:

python object_detection.py

