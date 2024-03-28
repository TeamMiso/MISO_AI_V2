from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from PIL import Image as PILImage
from io import BytesIO
from torchvision import transforms
from ultralytics import YOLO

from collections import OrderedDict

app = FastAPI()

class ImageRequest(BaseModel):
    imageURL: str

model = YOLO('best.pt')
names = model.names

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

def post_processing(out):
    name_cls = []
    for r in out:
        for c in r.boxes.cls:
            name_cls.append(names[int(c)])
    return name_cls

@app.post("/image-inference", response_model=List[str])
async def image_inference(request: ImageRequest):
    image_url = request.imageURL

    # requests를 사용하여 이미지 가져오기
    response = requests.get(image_url)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch image")

    image_bytes = response.content

    # 이미지 전처리
    img_tensor = preprocess_img(image_bytes)

    # 이미지 추론
    out = model.predict(img_tensor,save=True)
    result = post_processing(out)

    return list(OrderedDict.fromkeys(result))