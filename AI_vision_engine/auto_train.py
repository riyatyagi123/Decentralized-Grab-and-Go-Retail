import os
import cv2
import torch
import yaml
from ultralytics import YOLO
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
def auto_label_and_train():
    print(" AI Auto-Labeling System Started...")
    print("Loading Teacher Model (YOLO-World Zero-Shot)...")
    try:
        from ultralytics import YOLOWorld
        teacher_model = YOLOWorld("yolov8s-world.pt")
        teacher_model.set_classes(["maggie packet", "lays chips", "chocolate", "bottle", "grocery item"])
    except Exception as e:
        print("YOLOWorld not found, falling back to standard YOLOv8x...")
        teacher_model = YOLO("yolov8x.pt")
    img_dir = "custom_dataset/raw_images"
    train_images_dir = "custom_dataset/images/train"
    train_labels_dir = "custom_dataset/labels/train"
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    if not images:
        print(" No images found in custom_dataset/raw_images!")
        return
    print(f" Scanning and auto-labeling {len(images)} images using Zero-Shot Intelligence...")
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)
        new_img_path = os.path.join(train_images_dir, img_name)
        cv2.imwrite(new_img_path, frame)
        results = teacher_model.predict(frame, conf=0.10, verbose=False)
        label_file = os.path.join(train_labels_dir, img_name.replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            if results[0].boxes is not None:
                boxes = results[0].boxes.xywhn.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                for box, cls in zip(boxes, clss):
                    name = teacher_model.names[int(cls)]
                    if name != "person":
                        x_c, y_c, w, h = box
                        f.write(f"0 {x_c} {y_c} {w} {h}\n")
    print(" Auto-labeling complete!")
    yaml_content = {
        "path": os.path.abspath("custom_dataset"),
        "train": "images/train",
        "val": "images/train", 
        "names": {
            0: "Grocery Product"
        }
    }
    with open("custom_dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(" Starting Automated Self-Training!")
    student_model = YOLO("yolov8s.pt")
    try:
        student_model.train(
            data="custom_dataset.yaml",
            epochs=10,
            imgsz=640,
            batch=4,
            project="EdgeCart_Vision",
            name="auto_trained_model",
            device="0" if torch.cuda.is_available() else "cpu"
        )
        print(" AUTOMATED TRAINING COMPLETE!")
        print("Your new custom model is saved in: runs/detect/auto_trained_model/weights/best.pt")
    except Exception as e:
        print(f"Training Error: {e}")
if __name__ == "__main__":
    auto_label_and_train()
