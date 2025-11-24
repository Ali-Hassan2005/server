import google.generativeai as genai
import PIL.Image
import io
import os
import re
from fastapi import UploadFile, HTTPException
from typing import Dict, Optional

 
MODEL_NAME = "models/gemini-2.5-flash" 

 
_gemini_model = None

def _get_gemini_model():
    """
    Lazy load and configure Gemini model.
    Throws error immediately if API Key is missing.
    """
    global _gemini_model
    if _gemini_model is None:
        api_key = ''
        
        if not api_key or api_key.strip() == "":
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
            
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(MODEL_NAME)
        
    return _gemini_model

def _get_default_math_prompt() -> str:
    return """
You are a meticulous mathematical OCR engine. 
Your GOAL is to transcribe EVERY SINGLE equation in this image.

INSTRUCTIONS:
1. Output strictly in LaTeX format.
2. Put each equation on a new line inside $$ markers.
3. If there are conditions (like a ≠ 0), include them.
4. Do NOT include any conversational text, just the equations.

Output format example:
$$ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$
"""

def _parse_latex_equations(latex_text: str) -> list:
    if not latex_text:
        return []
   
    equations = re.findall(r'\$\$(.+?)\$\$', latex_text, re.DOTALL)
    
    return [eq.strip() for eq in equations if eq.strip()]

async def extract_math_equations(
    image: UploadFile,
    custom_prompt: Optional[str] = None
) -> Dict:
    """
    Extract mathematical equations from image using Google Gemini AI (Async).
    """
    try:
       
        await image.seek(0)
        contents = await image.read()
        
        if not contents:
            return {"success": False, "error": "File is empty"}

      
        try:
            img = PIL.Image.open(io.BytesIO(contents))
        except Exception:
             return {"success": False, "error": "Invalid image file format"}
 
        prompt = custom_prompt if custom_prompt else _get_default_math_prompt()
        
      
        try:
            model = _get_gemini_model()
        except ValueError as e:
            return {"success": False, "error": str(e)}

      
        response = await model.generate_content_async([prompt, img])
      
        if not response.candidates or not response.parts:
           
             finish_reason = response.prompt_feedback if hasattr(response, 'prompt_feedback') else "Unknown"
             return {
                 "success": False, 
                 "error": f"Model blocked the response. Reason: {finish_reason}",
                 "equation_count": 0
             }

      
        try:
            latex_text = response.text
        except ValueError:
           
            return {"success": False, "error": "Response was blocked by safety filters."}

        equations = _parse_latex_equations(latex_text)
        
        return {
            "success": True,
            "equation_count": len(equations),
            "equations": equations,
            "raw_output": latex_text
        }
        
    except Exception as e:
  
        print(f"Server Error in extract_math_equations: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "equations": [],
            "equation_count": 0
        }