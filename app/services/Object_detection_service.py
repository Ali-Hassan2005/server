from fastapi import UploadFile
from ultralytics import YOLO
from PIL import Image  
import io
from pathlib import Path

# Get the absolute path to the model file
MODEL_PATH = Path(__file__).parent.parent / "ai_models" / "yolov11.pt"
model = YOLO(str(MODEL_PATH))

 
NAMES = ["Headphones", "earbuds", "mobile phone", "sunglasses"]

def detect(file: UploadFile):
    
  
    if not file:
        return {"error": "No file provided"}

    try:
    
        image_bytes = file.file.read()
        img = Image.open(io.BytesIO(image_bytes))

  
        results = model(img)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])

            class_name = NAMES[cls_id] if cls_id < len(NAMES) else results.names[cls_id]

            detections.append({   
                "class_name": class_name,
            })

        return {
            "detections": detections,
        }

    except Exception as e:
        return {"error": str(e)}