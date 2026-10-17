import cv2
from ultralytics import YOLOWorld
model = YOLOWorld('yolov8l-world.pt')
model.set_classes(['bell pepper', 'vegetable', 'food', 'person'])
cap = cv2.VideoCapture('istockphoto-2218894868-640_adpp_is.mp4')
detects = set()
for _ in range(60):
    ret, frame = cap.read()
    if not ret: break
    res = model(frame, verbose=False)[0]
    for c, conf in zip(res.boxes.cls, res.boxes.conf):
        name = model.names[int(c)]
        if name != 'person': detects.add(f'{name} ({conf:.2f})')
print('YOLO-World Large detected:', detects)
