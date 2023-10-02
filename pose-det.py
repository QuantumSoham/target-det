from ultralytics import YOLO
import numpy
model=YOLO('./yolo weights/yolov8l-pose.pt')
results=model(source=0,show=True,conf=0.1,save=False)
# for r in results:
#     kpts=r.keypoints
#     print(kpts.cpu().numpy())