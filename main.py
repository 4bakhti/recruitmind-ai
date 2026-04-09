import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import uvicorn

# Import our Agent 01 logic
from agents.agent_01_parser import extract_text_from_pdf, parse_cv_with_gemini

app = FastAPI(
    title="RecruitMind AI API",
    description="Multi-Agent CV Intelligence System",
    version="1.0.0"
)

# Ensure a temporary directory exists to hold uploaded CVs
TEMP_DIR = "temp_cvs"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/health")
async def health_check():
    """Simple health check to ensure the API is running."""
    return {"status": "healthy", "system": "RecruitMind AI"}

@app.post("/api/v1/upload-cvs/")
async def upload_cvs(files: List[UploadFile] = File(...)):
    """
    Accepts bulk CV uploads, extracts text, and parses via Agent 01.
    """
    accepted_formats = [
        "application/pdf", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    
    processed_results = []
    
    for file in files:
        if file.content_type not in accepted_formats:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not supported.")
        
        # 1. Save the file temporarily
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Extract Text
        raw_text = extract_text_from_pdf(file_path)
        
        # 3. Parse with Gemini (Agent 01)
        if raw_text:
            parsed_data = parse_cv_with_gemini(raw_text)
            
            if parsed_data:
                processed_results.append({
                    "filename": file.filename,
                    "status": "success",
                    "data": parsed_data.model_dump() # Convert Pydantic model to dictionary
                })
            else:
                processed_results.append({
                    "filename": file.filename,
                    "status": "failed_parsing",
                    "error": "Gemini could not structure the data."
                })
        else:
            processed_results.append({
                "filename": file.filename,
                "status": "failed_extraction",
                "error": "Could not extract text from the PDF."
            })
            
        # Optional: Clean up the temp file after processing
        os.remove(file_path)

    return {
        "message": f"Processed {len(files)} CV(s).",
        "results": processed_results
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)