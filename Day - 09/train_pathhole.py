from ultralytics import YOLO
import torch

model = YOLO('yolov8n.pt')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    patience=20,
    project='runs',
    name='pathhole_yolov8n',
    device=device,
    amp=True,
)

print('Training finished.')
print('Best weights:', results.save_dir)
