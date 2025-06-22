from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO('runs/train_run_02/weights/best.pt')  

@app.get('/')
def hello():
    return {"message": "Hello from Helmet_detection with YOLO!"}

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)
        
    boxes = results[0].boxes
       
    helmet_count = len(boxes)

    confidences = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []

    return JSONResponse(content={
        "helmet_count": helmet_count,
        "confidences": confidences
    })