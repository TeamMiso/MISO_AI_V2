# service.py : model serving api

import numpy as np
import torch
from torchvision import transforms
import base64
from io import BytesIO
import bentoml
from PIL import Image as PILImage
import requests
from bentoml.io import DataFrame, JSON

yolov8m_runner = bentoml.pytorch.get("yolov8m_model").to_runner()
svc = bentoml.Service("yolov8_svc", runners=[yolov8m_runner])
class_names = {0: 'BAG', 1: 'BATTERY', 2: 'CAN', 3: 'CELL_PHONE', 4: 'CLOTHES', 5: 'GENERAL_TRASH', 6: 'GLASSES', 7: 'LAPTOP', 8: 'LDPE', 9: 'LIGHTER', 10: 'MOUSE', 11: 'PAPER', 12: 'PAPER_PACK', 13: 'PET', 14: 'PLASTIC', 15: 'SHOES', 16: 'TOOTHBRUSH', 17: 'VINYL'}

def preprocess_img(imageURL):
    image = PILImage.open(BytesIO(imageURL))

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        
    # 입력 이미지의 데이터 타입을 torch.FloatTensor로 변환
    image = transform(image).float().unsqueeze(0)

def post_processing(img_tensor, out):
    cls_names = []
    for r in out:
        for c in r.boxes.cls:
            cls_names.append(class_names[int(c)])

    return cls_names

@svc.api(input=FormDataInput(), output=JSON())
def imageInference(imageURL: str):
    response = requests.get(imageURL)
    bytesImage = response.content

    img_tensor = preprocess_img(bytesImage)

    # image inference
    out = yolov8m_runner.run(img_tensor)

    result_class_list = post_processing(img_tensor, out)

    return result_class_list
