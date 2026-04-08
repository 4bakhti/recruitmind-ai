from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import uvicorn

app = FastAPI(
    title="RecruitMind AI API",
    description="Multi-Agent CV Intelligence System",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Simple health check to ensure the API is running."""
    return {"status": "healthy", "system": "RecruitMind AI"}

@app.post("/api/v1/upload-cvs/")
async def upload_cvs(files: List[UploadFile] = File(...)):
    """
    Accepts bulk CV uploads (PDF, DOCX) to trigger the pipeline.
    """
    accepted_formats = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    
    processed_files = []
    
    for file in files:
        if file.content_type not in accepted_formats:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a supported format. Please upload PDF or DOCX.")
        
        # Here we will eventually save the file temporarily 
        # and pass the file path or raw bytes to the Orchestrator Agent
        
        processed_files.append({"filename": file.filename, "status": "queued_for_processing"})

    # TODO: Trigger LangGraph Orchestrator here
    
    return {
        "message": f"Successfully queued {len(files)} CVs for processing.",
        "batch_details": processed_files
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)