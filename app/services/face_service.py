import face_recognition
import numpy as np
from fastapi import UploadFile
from app.utils.image_utils import uploadfile_to_numpy

def get_face_encoding(image_np: np.ndarray):
    encodings = face_recognition.face_encodings(image_np)
    if not encodings:
        return None
    return encodings[0]

def is_match(ref: UploadFile, test: UploadFile, tolerance=0.6) -> dict:

    ref_image_np = uploadfile_to_numpy(ref)
    ref_encoding = get_face_encoding(ref_image_np)
    if ref_encoding is None:
        return {"match": False, "error": "No face found in reference image"}

    test_image_np = uploadfile_to_numpy(test)  
    test_encoding = get_face_encoding(test_image_np)
    if test_encoding is None:
        return {"match": False, "error": "No face found in frame"}

    matches = face_recognition.compare_faces([ref_encoding], test_encoding, tolerance=tolerance)
    
    return {
        "match": bool(matches[0]),
        "error": None
    }


def count_faces_in_frame(image: UploadFile, model="hog") -> int:
    image_np = uploadfile_to_numpy(image)
    face_locations = face_recognition.face_locations(image_np, model=model)
    return len(face_locations)
