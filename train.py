import torch
import torchvision
from ultralytics import YOLO
import os # Import os module for checking if running in a frozen environment

if __name__ == '__main__':
    # Add this line specifically for Windows/frozen environments
    # It's good practice for multiprocessing on Windows
    # from multiprocessing import freeze_support # You don't strictly need this unless freezing to an executable
    # freeze_support() # Call freeze_support() if you were building an executable

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n model

    # Use the model
    results = model.train(data='data.yaml', epochs=100, imgsz=640, batch=16, device=0, name='yolov8n_custom')