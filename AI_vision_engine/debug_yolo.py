import cv2
from ultralytics import YOLO
model = YOLO('yolov8x.pt')
cap = cv2.VideoCapture('istockphoto-2218894868-640_adpp_is.mp4')
classes = set()
for _ in range(120):
    ret, frame = cap.read()
    if not ret: break
    res = model(frame, verbose=False)[0]
    for c, conf in zip(res.boxes.cls, res.boxes.conf):
        name = model.names[int(c)]
        if name != 'person' and conf > 0.15:
            classes.add(name)
print('DETECTED IN VIDEO:', classes)
