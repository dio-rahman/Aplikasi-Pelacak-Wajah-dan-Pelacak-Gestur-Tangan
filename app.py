# main.py
from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os

app = FastAPI()
upload_router = APIRouter()

# Ensure upload directory exists
os.makedirs("static/uploads", exist_ok=True)

@upload_router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    filename = f"{uuid4()}.png"
    file_path = os.path.join("static/uploads", filename)
    cv2.imwrite(file_path, img)
    return {
        "image_path": f"/static/uploads/{filename}",
        "rgb_array": {"R": r.tolist(), "G": g.tolist(), "B": b.tolist()}
    }

app.include_router(upload_router)
# Add other routers for remaining features here