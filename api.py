from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the YOLO model (make sure 'best.pt' is in your project directory)
model = YOLO('runs/train_run_2/weights/best.pt')  # ✅ forward slashes



@app.get('/')
def hello():
    return {"message": "Hello from Helmet_detection with YOLO!"}


@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    results = model(image)
    boxes = results[0].boxes

    # Check if boxes are detected
    if boxes is None or len(boxes) == 0:
        return JSONResponse(content={"message": "No objects detected."})

    result_json = results[0].to_json()  # Corrected method
    return JSONResponse(content={"predictions": result_json})
