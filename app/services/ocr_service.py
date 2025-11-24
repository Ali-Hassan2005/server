import re
import string
import numpy as np
import cv2
from paddleocr import PaddleOCR
from fastapi import UploadFile
from typing import Dict, List, Optional

# Global OCR model (lazy loading)
_ocr_model = None


def _get_ocr_model(lang='en', use_angle_cls=True):
    """Lazy load OCR model."""
    global _ocr_model
    if _ocr_model is None:
        _ocr_model = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)
    return _ocr_model


def preprocess_text(text: str) -> str:
    """Clean and normalize extracted text."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub("\s+", " ", text).strip()
    return text


async def extract_text_from_image(image: UploadFile, preprocess: bool = False, lang: str = 'en') -> Dict:
    """
    Extract text from uploaded image using PaddleOCR.
    
    Args:
        image: FastAPI UploadFile containing the image
        preprocess: Whether to clean/normalize the extracted text
        lang: OCR language (default: 'en')
        
    Returns:
        dict: Contains extracted text and metadata
    """
    try:
        # Read image from upload
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "error": "Failed to decode image",
                "text": "",
                "lines": []
            }
        
        # Perform OCR
        ocr_model = _get_ocr_model(lang=lang)
        result = ocr_model.ocr(img, cls=True)
        
        if not result or not result[0]:
            return {
                "text": "",
                "lines": [],
                "word_count": 0
            }
        
        # Extract text lines with confidence
        text_lines = []
        detailed_results = []
        
        for line in result[0]:
            bbox = line[0]  # Bounding box coordinates
            text_info = line[1]  # (text, confidence)
            text = text_info[0]
            confidence = text_info[1]
            
            text_lines.append(text)
            detailed_results.append({
                "text": text,
                "confidence": round(confidence, 4),
                "bbox": bbox
            })
        
        # Join all text
        full_text = "\n".join(text_lines)
        
        # Apply preprocessing if requested
        if preprocess:
            full_text = preprocess_text(full_text)
        
        return {
            "text": full_text,
            "lines": text_lines,
            "word_count": len(full_text.split()),
            "line_count": len(text_lines),
            "detailed_results": detailed_results
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "text": "",
            "lines": []
        }
