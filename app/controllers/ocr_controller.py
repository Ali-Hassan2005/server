from fastapi import UploadFile
from app.services.ocr_service import extract_text_from_image
from app.services.math_ocr_service import extract_math_equations
from typing import Optional


class OCRController:
    """Controller for OCR operations."""
    
    async def extract_text(self, image: UploadFile, preprocess: bool = False, lang: str = 'en'):
        """
        Extract text from image using PaddleOCR.
        
        Args:
            image: Image file containing text
            preprocess: Clean and normalize text
            lang: OCR language
            
        Returns:
            dict: Extracted text and metadata
        """
        try:
            result = await extract_text_from_image(image, preprocess=preprocess, lang=lang)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "text": "",
                "lines": []
            }
    
    async def extract_math(self, image: UploadFile, custom_prompt: Optional[str] = None):
        """
        Extract mathematical equations from image using Gemini AI.
        
        Args:
            image: Image file containing mathematical equations
            custom_prompt: Custom prompt for specific extraction needs
            
        Returns:
            dict: Extracted LaTeX equations and metadata
        """
        try:
            result = await extract_math_equations(image, custom_prompt=custom_prompt)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "raw_output": "",
                "equations": [],
                "equation_count": 0,
                "success": False
            }


# Singleton instance
ocr_controller = OCRController()
