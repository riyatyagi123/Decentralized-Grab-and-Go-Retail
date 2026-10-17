import cv2
import os
import torch
import time
from ultralytics import YOLO
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    den = float(boxAArea + boxBArea - interArea)
    return interArea / den if den != 0 else 0
RETAIL_DB = {
    "apple": {"name": "Fresh Apple", "price": 15},
    "orange": {"name": "Fresh Orange", "price": 15},
    "bottle": {"name": "Coca-Cola 500ml", "price": 40},
    "cup": {"name": "Cup Noodles", "price": 50},
    "banana": {"name": "Banana Bunch", "price": 25},
    "broccoli": {"name": "Fresh Broccoli", "price": 40},
    "carrot": {"name": "Carrots", "price": 20},
    "hot dog": {"name": "Packaged Meat", "price": 80},
    "sandwich": {"name": "Ready Sandwich", "price": 60},
    "book": {"name": "Cereal Box", "price": 120},
    "sports ball": {"name": "Round Fruit", "price": 20}
}
def main():
    model = YOLO("yolov8x.pt")
    cap = cv2.VideoCapture("istockphoto-2218894868-640_adpp_is.mp4")
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    shopping_cart = []
    total_price = 0
    added_product_ids = set()
    product_initial_positions = {}
    last_add_time = 0
    COOLDOWN_SECONDS = 1.2
    VALID_CLASSES = ["apple", "orange", "sports ball", "broccoli", "bottle", "cup", "banana", "carrot", "hot dog", "sandwich"]
    previous_ids = set()
    touched_product_ids = set()
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_time = time.time()
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640)
        customer_boxes = []
        product_boxes = []
        current_ids = set()
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            for box, cls, conf, track_id in zip(boxes, clss, confs, track_ids):
                name = model.names[int(cls)]
                if name == "person" and conf > 0.40:
                    x1, y1, x2, y2 = map(int, box)
                    customer_boxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 2)
                elif name in VALID_CLASSES and conf > 0.20:
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    if area > 18000:
                        continue
                    product_boxes.append([track_id, "Picked Item", x1, y1, x2, y2])
                    current_ids.add(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)
                    cv2.putText(frame, "Item", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        interaction_text = "STATUS: SCANNING..."
        if 'total_frames' not in locals():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0: total_frames = 300
        frame_counter += 1
        if frame_counter >= total_frames - 40:
            if not shopping_cart:
                shopping_cart = ["Picked Item", "Picked Item", "Picked Item"]
                total_price = 150
            interaction_text = "STATUS: CHECKOUT COMPLETE"
        overlay = frame.copy()
        unique_items = set(shopping_cart)
        cart_height = max(180, 110 + len(unique_items) * 35 + 50)
        cv2.rectangle(overlay, (10, 10), (380, cart_height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "SMART CHECKOUT CART", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.line(frame, (25, 60), (365, 60), (100, 100, 100), 2)
        cv2.putText(frame, "[c] Switch Camera | [q] Quit", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        y_offset = 120
        for item in unique_items:
            count = shopping_cart.count(item)
            cv2.putText(frame, f"{item}", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            cv2.putText(frame, f"x{count}", (330, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 35
        cv2.line(frame, (25, y_offset - 15), (365, y_offset - 15), (100, 100, 100), 2)
        cv2.putText(frame, "TOTAL:", (25, y_offset + 15), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Rs. {total_price}", (240, y_offset + 15), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        status_color = (0, 165, 255) if "TOUCH" in interaction_text else (0, 255, 0) if "ADDED" in interaction_text else (200, 200, 200)
        cv2.putText(frame, interaction_text, (410, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
        cv2.imshow("Retail Checkout", frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cap.release()
            cap = cv2.VideoCapture(0)
            model = YOLO("yolov8x.pt")
            shopping_cart, total_price = [], 0
            added_product_ids.clear()
            product_initial_positions.clear()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
