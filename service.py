# service.py : model serving api
import io
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import base64
from io import BytesIO
from PIL import Image as PILImage
import requests
import pandas as pd

import bentoml
from bentoml.io import JSON

class_names = {
        0: 'BAG', 1: 'BATTERY', 2: 'CAN', 3: 'CELL_PHONE', 4: 'CLOTHES',
        5: 'GENERAL_TRASH', 6: 'GLASSES', 7: 'LAPTOP', 8: 'LDPE', 9: 'LIGHTER',
        10: 'MOUSE', 11: 'PAPER', 12: 'PAPER_PACK', 13: 'PET', 14: 'PLASTIC',
        15: 'SHOES', 16: 'TOOTHBRUSH', 17: 'VINYL'
    }
yolov8m_runner = bentoml.pytorch.get("yolov8m_model").to_runner()
svc = bentoml.Service("yolov8_svc", runners=[yolov8m_runner])

def preprocess_img(imageBytes):
    image = PILImage.open(BytesIO(imageBytes))

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        
    # 입력 이미지의 데이터 타입을 torch.FloatTensor로 변환
    image = transform(image).float().unsqueeze(0)
    
    return image

@svc.api(input=JSON(), output=JSON())
def imageInference(imageURL: str) -> list:
    image_url = imageURL.get('imageURL')

    # requests를 사용하여 이미지 가져오기
    response = requests.get(image_url)
    image_bytes = response.content

    # 이미지 전처리
    img_tensor = preprocess_img(image_bytes)

    # 이미지 추론
    out = yolov8m_runner.run(img_tensor)
    return out
