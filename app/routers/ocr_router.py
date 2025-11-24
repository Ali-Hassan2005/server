from fastapi import APIRouter, UploadFile, File, Query
from app.controllers.ocr_controller import ocr_controller
from typing import Optional

router = APIRouter()


@router.post("/text-extraction")
async def extract_text_endpoint(
    image: UploadFile = File(..., description="Image containing text to extract"),
    preprocess: bool = Query(False, description="Clean and normalize extracted text"),
    lang: str = Query('en', description="OCR language (en, ar, ch, etc.)")
):
 
    return await ocr_controller.extract_text(image, preprocess=preprocess, lang=lang)


@router.post("/math-extraction")
async def extract_math_endpoint(
    image: UploadFile = File(..., description="Image containing mathematical equations"),
    custom_prompt: Optional[str] = Query(None, description="Custom prompt for specific needs")
):
 
    return await ocr_controller.extract_math(image, custom_prompt=custom_prompt)


@router.post("/combined-extraction")
async def combined_extraction_endpoint(
    image: UploadFile = File(..., description="Image containing both text and equations"),
    lang: str = Query('en', description="Language for text OCR")
):
 
    await image.seek(0)
    text_result = await ocr_controller.extract_text(image, preprocess=False, lang=lang)
    
    await image.seek(0)
    math_result = await ocr_controller.extract_math(image)
    
    return {
        "text_extraction": text_result,
        "math_extraction": math_result
    }
