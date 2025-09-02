from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
import os
import uuid

# Safe unpickling
torch.serialization.add_safe_globals([DetectionModel])

app = FastAPI()

# Load YOLOv8 model
model = YOLO("best.pt")

# Create directories
UPLOAD_DIR = "uploads"
RESULT_DIR = "runs/detect"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Serve static files
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
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Run YOLOv8 prediction
    results = model.predict(source=file_path, save=True, imgsz=640)

    # Get the folder where results are saved
    result_folder = os.path.basename(results[0].save_dir)
    result_dir = results[0].save_dir

    # Find the actual image file saved by YOLO
    saved_images = [f for f in os.listdir(result_dir) if f.endswith(".jpg")]
    if not saved_images:
        return """
        <html>
            <body>
                <h2>No result image found.</h2>
                <a href="/">Go back</a>
            </body>
        </html>
        """

    result_img_name = saved_images[0]  # Use the first .jpg found
    result_img_path = f"/runs/detect/{result_folder}/{result_img_name}"

    # Return HTML with correct image path
    return f"""
    <html>
        <head><title>Detection Result</title></head>
        <body>
            <h2>Detection Result</h2>
            <img src="{result_img_path}" alt="Result">
            <br><a href="/">Try another</a>
        </body>
    </html>
    """
