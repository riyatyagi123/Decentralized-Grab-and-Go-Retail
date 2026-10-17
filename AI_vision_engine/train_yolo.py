import os
import torch
from ultralytics import YOLO
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
def train_perfect_yolo():
    print(" Initializing Decentralized Grab-and-Go Vision Training...")
    model = YOLO("yolov8n.pt") 
    training_args = {
        "data": "dataset.yaml",
        "epochs": 300,
        "patience": 50,
        "imgsz": 640,
        "batch": 16,
        "device": "0" if torch.cuda.is_available() else "cpu",
        "optimizer": "auto",
        "lr0": 0.001,
        "augment": True,
        "mosaic": 1.0,
        "project": "EdgeCart_Vision",
        "name": "yolov8n_hand_product"
    }
    try:
        results = model.train(**training_args)
        exported_path = model.export(format="onnx", optimize=True)
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    train_perfect_yolo()
