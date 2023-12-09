from ultralytics import YOLO
import torch

print(torch.cuda.is_available())

model_path = './runs/detect/train/weights/best.pt'

# Initialize
model = YOLO(model_path)  # or yolov5m, yolov5l, yolov5x, custom

result = model(source="vid.mp4", show=True, conf=0.3) 
