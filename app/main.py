from fastapi import FastAPI
from app.routers.proctoring_router import router as proctoring_router
from app.routers.ocr_router import router as ocr_router
 

app = FastAPI(
    title="AI Proctoring & OCR Server",
    description="Advanced AI server for exam proctoring and OCR services",
    version="1.0.0"
)

app.include_router(proctoring_router, prefix="/api/proctoring", tags=["Proctoring"])
app.include_router(ocr_router, prefix="/api/ocr", tags=["Proctoring"])
 
 

@app.get("/")
def root():
    return {"message": "AI Server Running!"}
