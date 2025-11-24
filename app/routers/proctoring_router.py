from fastapi import APIRouter, UploadFile, File
from app.controllers.proctoring_controller import proctoring_controller
from app.services.gaze_attention_service import gaze_service
from typing import Optional

router = APIRouter()

@router.post("/detect-cheating")
async def detect_cheating(
    image: UploadFile = File(..., description="Current frame to analyze"),
    reference_image: Optional[UploadFile] = File(None, description="Reference image for face matching")
):
    return await proctoring_controller.detect_cheating(image, reference_image)
 
@router.post("/detect")
async def detect_attention(
    image: UploadFile = File(..., description="Current frame to analyze"),
):
    return await gaze_service.analyze_attention(image)
 
