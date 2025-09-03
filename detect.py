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

# Home page
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>YOLOv8 Detection</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 flex items-center justify-center min-h-screen">
            <div class="bg-white shadow-lg rounded-2xl p-8 max-w-lg w-full text-center">
                <h2 class="text-2xl font-bold text-violet-600 mb-6">Upload an Image for Detection</h2>
                <form action="/predict" enctype="multipart/form-data" method="post" class="space-y-4">
                    <input name="file" type="file" accept="image/*" 
                        class="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4
                               file:rounded-full file:border-0 file:text-sm file:font-semibold
                               file:bg-violet-100 file:text-violet-700 hover:file:bg-violet-200"/>
                    <button type="submit" 
                        class="bg-violet-600 hover:bg-violet-700 text-white font-bold py-2 px-6 rounded-xl shadow-md transition">
                        Detect
                    </button>
                </form>
            </div>
        </body>
    </html>
    """

# Prediction endpoint
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
            <body class="bg-gray-100 flex items-center justify-center min-h-screen">
                <div class="bg-white shadow-lg rounded-2xl p-8 max-w-lg w-full text-center">
                    <h2 class="text-xl font-bold text-red-500 mb-4">No result image found.</h2>
                    <a href="/" class="text-violet-600 hover:underline">Go back</a>
                </div>
            </body>
        </html>
        """

    result_img_name = saved_images[0]  # Use the first .jpg found
    result_img_path = f"/runs/detect/{result_folder}/{result_img_name}"

    # Return styled result page
    return f"""
    <html>
        <head>
            <title>Detection Result</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 flex items-center justify-center min-h-screen">
            <div class="bg-white shadow-lg rounded-2xl p-8 max-w-xl w-full text-center">
                <h2 class="text-2xl font-bold text-violet-600 mb-6">Detection Result</h2>
                <img src="{result_img_path}" alt="Result" class="rounded-xl shadow-md mx-auto mb-6"/>
                <a href="/" class="bg-violet-600 hover:bg-violet-700 text-white font-bold py-2 px-6 rounded-xl shadow-md transition">
                    Try Another
                </a>
            </div>
        </body>
    </html>
    """
