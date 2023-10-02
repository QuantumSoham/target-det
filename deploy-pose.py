import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
import tempfile

# Load YOLO model
model = YOLO('yolo weights/yolov8l-pose.pt')

# Define class names
classname = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a Streamlit app
def main():
    st.title("YOLO Object Detection with Streamlit")
    video_file = st.file_uploader("people.mp4", type=["mp4"])

    if video_file is not None:
        cap = cv2.VideoCapture("people.mp4")

        while True:
            success, img = cap.read()
            if not success:
                st.warning("Video has ended.")
                break

            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                kpt = r.keypoints.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    cvzone.putTextRect(img, f'{classname[cls]} {int(conf * 100)}%', (max(0, x1), max(35, y1 - 20)))
                    st.image(img, caption=f'{classname[cls]} {int(conf * 100)}%', channels="BGR")

            st.write(img)

if __name__ == '__main__':
    main()