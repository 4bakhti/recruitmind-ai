import os
import shutil
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import ValidationError
from typing import List
import uvicorn

# --- Import our Agents ---
from agents.agent_01_parser import extract_text_from_pdf, parse_cv_with_gemini
from agents.agent_02_crawler import enrich_candidate_profile
from agents.agent_03_fusion import clean_and_fuse_profile
from agents.parser_agent import parse_cv as parse_cv_with_claude
from agents.github_crawler_agent import fetch_github_profile
from models.job_description import JobDescription
from orchestrator import run_pipeline

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
    return {"status": "healthy", "system": "RecruitMind AI - Phase 1 Pipeline"}

@app.post("/api/v1/upload-cvs/")
async def upload_cvs(files: List[UploadFile] = File(...)):
    """
    Accepts bulk CV uploads and runs them through Agents 01, 02, and 03.
    """
    accepted_formats = [
        "application/pdf", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    
    processed_results = []
    
    for file in files:
        if file.content_type not in accepted_formats:
            processed_results.append({
                "filename": file.filename,
                "status": "error",
                "error": "Unsupported file format. Please upload a PDF."
            })
            continue
        
        # 1. Save the file temporarily
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            print(f"\n==============================================")
            print(f"🚀 STARTING PIPELINE FOR: {file.filename}")
            print(f"==============================================")
            
            # --- AGENT 01: Parse & Extract ---
            print("[Agent 01] Extracting text and parsing with Gemini...")
            raw_text = extract_text_from_pdf(file_path)
            if not raw_text:
                raise ValueError("Could not extract text from the PDF.")
                
            candidate = parse_cv_with_gemini(raw_text)
            if not candidate:
                raise ValueError("Agent 01 failed to structure the CV data.")

            # --- AGENT 02: Profile Crawler ---
            # Note the 'await' here because Agent 02 is asynchronous!
            if candidate.github_url:
                print(f"[Agent 02] GitHub URL found. Crawling metrics for {candidate.full_name}...")
                candidate = await enrich_candidate_profile(candidate)
            else:
                print("[Agent 02] No GitHub URL found. Skipping crawler.")

            # --- AGENT 03: Data Fusion & Cleaner ---
            print(f"[Agent 03] Cleaning data, normalizing skills, and checking gaps...")
            final_candidate = clean_and_fuse_profile(candidate)

            print(f"✅ PIPELINE COMPLETE FOR: {file.filename}\n")

            # Append successful result
            processed_results.append({
                "filename": file.filename,
                "status": "success",
                "data": final_candidate.model_dump()
            })

        except Exception as e:
            print(f"❌ Error processing {file.filename}: {e}")
            processed_results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
            
        finally:
            # Always clean up the temporary file, even if the pipeline crashes
            if os.path.exists(file_path):
                os.remove(file_path)

    return {
        "message": f"Processed {len(files)} CV(s).",
        "results": processed_results
    }

def _save_upload(file: UploadFile) -> str:
    """Validate a CV upload (PDF/DOCX) and save it to a temp path."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF or DOCX file.",
        )
    file_path = os.path.join(TEMP_DIR, f"claude_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


def _run_or_422(file_path: str, job: JobDescription | None = None) -> dict:
    """Run the orchestrated pipeline, or fail the request if the CV could
    not be parsed at all. Every later stage degrades gracefully instead,
    recording why in the returned state."""
    state = run_pipeline(file_path, job)
    if state.get("candidate") is None:
        detail = "; ".join(state.get("errors", [])) or (
            "Could not parse the CV. Check the server logs for details."
        )
        raise HTTPException(status_code=422, detail=detail)
    return state


@app.post("/parse-cv")
def parse_cv_endpoint(file: UploadFile = File(...)):
    """
    Parse a single CV (PDF or DOCX) with the Claude-based parser agent
    and return the structured Candidate JSON. Standalone — does not run
    the crawler/fusion pipeline.
    """
    file_path = _save_upload(file)
    try:
        candidate = parse_cv_with_claude(file_path)
        if candidate is None:
            raise HTTPException(
                status_code=422,
                detail="Could not parse the CV. Check the server logs for details.",
            )
        return candidate.model_dump()
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/analyze-candidate")
def analyze_candidate_endpoint(file: UploadFile = File(...)):
    """
    The full Claude pipeline: parse the CV, enrich with GitHub data, and
    fuse into one annotated Candidate (normalized skills, inferred
    GitHub skills, employment-gap notes, data-quality notes).
    """
    file_path = _save_upload(file)
    try:
        state = _run_or_422(file_path)
        return state["candidate"].model_dump(mode="json")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/score-fit")
def score_fit_endpoint(
    file: UploadFile = File(...),
    job_description: str = Form(
        ...,
        description="JobDescription as a JSON string, e.g. "
        '{"title": "Backend Engineer", "required_skills": ["Python"], '
        '"nice_to_have_skills": [], "description_text": "..."}',
    ),
):
    """
    Full pipeline + scoring: parse -> GitHub enrich -> fuse -> score the
    candidate against the given job description. Returns the annotated
    Candidate and the JobFitResult together.

    Note: because the CV arrives as a multipart file upload, the job
    description is sent as a JSON string in the `job_description` form
    field (multipart and a raw JSON body cannot be mixed in one request).
    """
    try:
        job = JobDescription.model_validate_json(job_description)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid job_description JSON: {e.error_count()} "
                   f"validation error(s): {e}",
        )

    file_path = _save_upload(file)
    try:
        state = _run_or_422(file_path, job)
        if state.get("fit") is None:
            raise HTTPException(
                status_code=502,
                detail="; ".join(state.get("errors", [])) or "Scoring failed.",
            )
        return {
            "candidate": state["candidate"].model_dump(mode="json"),
            "job_fit": state["fit"].model_dump(mode="json"),
        }
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/full-report")
def full_report_endpoint(
    file: UploadFile = File(...),
    job_description: str = Form(
        ...,
        description="JobDescription as a JSON string (same shape as /score-fit)",
    ),
):
    """
    The complete pipeline: parse -> GitHub enrich -> fuse -> job-fit
    scoring -> report generation. Returns the CandidateReport as JSON,
    including the rendered Markdown in `report_markdown`.
    """
    try:
        job = JobDescription.model_validate_json(job_description)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid job_description JSON: {e.error_count()} "
                   f"validation error(s): {e}",
        )

    file_path = _save_upload(file)
    try:
        state = _run_or_422(file_path, job)
        if state.get("report") is None:
            raise HTTPException(
                status_code=502,
                detail="; ".join(state.get("errors", []))
                or "Report generation failed.",
            )
        return state["report"].model_dump(mode="json")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/crawl-github")
def crawl_github_endpoint(github_username: str):
    """
    Fetch a GitHub profile summary for a username via the GitHub Crawler
    Agent. Standalone endpoint for testing the agent independently of the
    full pipeline.
    """
    profile = fetch_github_profile(github_username)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch GitHub profile for '{github_username}' "
                   "(unknown user, rate limit, or network error — see logs).",
        )
    return profile.model_dump(mode="json")


@app.post("/run-pipeline")
def run_pipeline_endpoint(
    file: UploadFile = File(...),
    job_description: str | None = Form(
        None,
        description="Optional JobDescription as a JSON string. Omit it to "
        "stop after fusion and get just the annotated candidate.",
    ),
):
    """
    The orchestrated pipeline with its full trace exposed.

    Same work as /full-report, but the response says which nodes ran and
    what went wrong, and a partial run still returns whatever it produced
    instead of an error. Useful for debugging a CV that comes back thin.
    """
    job = None
    if job_description:
        try:
            job = JobDescription.model_validate_json(job_description)
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid job_description JSON: {e.error_count()} "
                       f"validation error(s): {e}",
            )

    file_path = _save_upload(file)
    try:
        state = _run_or_422(file_path, job)
        return {
            "candidate": state["candidate"].model_dump(mode="json"),
            "job_fit": (
                state["fit"].model_dump(mode="json")
                if state.get("fit") is not None else None
            ),
            "report": (
                state["report"].model_dump(mode="json")
                if state.get("report") is not None else None
            ),
            "steps_completed": state.get("steps_completed", []),
            "errors": state.get("errors", []),
        }
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)