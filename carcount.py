from ultralytics import YOLO
import cv2
import scipy
import cvzone
cap = cv2.VideoCapture("cars.mp4")
cap.set(3,1920)
cap.set(4,1080)
model = YOLO('yolo weights/yolov8l.pt')
while True:
    success, img=cap.read()
    results=model(img, stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,200,200),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)




