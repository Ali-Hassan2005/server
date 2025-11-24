from fastapi import UploadFile
from PIL import Image
from io import BytesIO
import numpy as np

def uploadfile_to_numpy(file: UploadFile) -> np.ndarray:
    image_bytes = file.file.read()
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB") 
    np_image = np.array(image, dtype=np.uint8)  
    return np_image