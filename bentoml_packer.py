import bentoml
from ultralytics import YOLO

model = YOLO("best.pt").model
model.eval()
saved_model = bentoml.pytorch.save_model(name='yolov8m_model', model=model, signatures={"__call__": {"batchable": False}})