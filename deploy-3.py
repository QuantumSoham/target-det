import streamlit as st
import pandas as pd
from ultralytics import YOLO
import cv2
import cvzone
import math

# Create the Streamlit website
def website():
    st.set_page_config(page_title='Govt of India', page_icon='6.png', layout='wide')
    st.image('6.png', width=100)
    st.markdown("<h1 style='text-align: center; font-size: 36px;'>Govt of India</h1>", unsafe_allow_html=True)
    st.title(" ")
    st.markdown("<h2 style='text-align: center; font-size: 24px;'>HUMAN TARGET ACQUISITION SYSTEM</h2>",
                unsafe_allow_html=True)
    st.title(" ")
    st.sidebar.header('Govt of India')
    st.title(" ")
    st.title(" ")
    st.sidebar.header('Ministry Of Defence')

    # User input features
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features

    st.image(['5.png'], width=100)
    st.markdown("<h3 style='text-align: center; font-size: 18px;'>Welcome to the Govt of India website!</h3>",
                unsafe_allow_html=True)
    st.image('3.png')
    st.title(" ")
    st.title(" ")

if __name__ == '__main__':
    website()

# Load YOLO model
model = YOLO('yolo weights/yolov8l-pose.pt')

# Define class names
classname = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a Streamlit app
def main():
    st.title("YOLO Object Detection .")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])

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
                    st.image(img, caption=f'{classname[cls]} {int(conf * 100)}% walking', channels="BGR")

            st.write(img)

if __name__ == '__main__':
    main()