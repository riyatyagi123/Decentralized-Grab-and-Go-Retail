import os
import random
from ultralytics import YOLO

def prepare_and_split_data(image_folder):
    all_images = os.listdir(image_folder)
    random.shuffle(all_images)
    
    total_images = len(all_images)
    train_split = int(total_images * 0.8)
    val_split = int(total_images * 0.9)
    
    training_data = all_images[:train_split]
    validation_data = all_images[train_split:val_split]
    testing_data = all_images[val_split:]
    
    return training_data, validation_data, testing_data

def execute_training():
    model = YOLO("yolov8x.pt")
    
    model.train(
        data="custom_dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="EdgeCart_Vision",
        name="retail_model",
        device="0"
    )

def execute_validation_and_testing():
    model = YOLO("EdgeCart_Vision/retail_model/weights/best.pt")
    
    validation_metrics = model.val(
        data="custom_dataset.yaml",
        split="val",
        plots=True,
        conf=0.25,
        iou=0.60,
        save_json=True
    )
    
    testing_metrics = model.val(
        data="custom_dataset.yaml",
        split="test",
        plots=True,
        conf=0.25
    )
    
    conf_matrix = validation_metrics.confusion_matrix
    conf_matrix.plot(save_dir="EdgeCart_Vision/evaluation_reports/", names=model.names)

if __name__ == "__main__":
    execute_training()
    execute_validation_and_testing()
