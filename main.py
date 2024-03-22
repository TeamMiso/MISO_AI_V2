from fastapi import FastAPI, HTTPException, UploadFile, File
from io import BytesIO
import numpy as np
import torch
from torchvision import transforms
import base64
import requests
from PIL import Image
import uvicorn
from ultralytics import YOLO

app = FastAPI()

model = YOLO('best.pt')
class_names = model.names

class imageToName:
    def __init__(self, image):
        self.image = image
    def preprocess_img(self):
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.image = Image.open(BytesIO(self.image))
        
        # 입력 이미지의 데이터 타입을 torch.FloatTensor로 변환
        self.image = transform(self.image).float().unsqueeze(0)

    def inference(self):
        with torch.no_grad():
            result = model.predict(self.image, save=True)
            cls_names = []
            for r in result:
                for c in r.boxes.cls:
                    cls_names.append(class_names[int(c)])
        
        return cls_names

@app.post("/postToImage")
async def postToImage(imageURL: str):
    try:
        
        response = requests.get(imageURL)
        img = response.content

        imgToCls = imageToName(img)

        imgToCls.preprocess_img()

        output = imgToCls.inference()

        return output
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))