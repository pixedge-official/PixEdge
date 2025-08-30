from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
import os
import uuid

# Safe unpickling
torch.serialization.add_safe_globals([DetectionModel])

app = FastAPI()
model = YOLO("best.pt")

UPLOAD_DIR = "uploads"
RESULT_DIR = "runs/detect"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/runs", StaticFiles(directory="runs"), name="runs")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>YOLOv8 Detection</title></head>
        <body>
            <h2>Upload an image for detection</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Detect">
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    results = model.predict(source=file_path, save=True, imgsz=640)
    result_img = os.path.join(results[0].save_dir, "predictions.jpg")

    return f"""
    <html>
        <head><title>Detection Result</title></head>
        <body>
            <h2>Detection Result</h2>
            <img src="/runs/detect/{os.path.basename(results[0].save_dir)}/predictions.jpg" alt="Result">
            <br><a href="/">Try another</a>
        </body>
    </html>
    """
