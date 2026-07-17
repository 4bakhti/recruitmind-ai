import json
from google import genai
from pydantic import ValidationError

from core.config import settings
from models.candidate import CandidateSchema

# Initialize Gemini Client safely
client = genai.Client(api_key=settings.GEMINI_API_KEY)

def clean_and_fuse_profile(candidate: CandidateSchema) -> CandidateSchema:
    """
    Acts as Agent 03: Normalizes skills, flags employment gaps, 
    and cleans up any messy data from the extraction phase.
    """
    print(f"Agent 03: Fusing and cleaning data for {candidate.full_name}...")
    
    # We turn the current candidate state into JSON so Gemini can read it
    candidate_json = candidate.model_dump_json()
    schema_instructions = json.dumps(CandidateSchema.model_json_schema())

    prompt = f"""
    You are an expert Data Cleaner AI for a recruitment pipeline.
    Take the provided RAW CANDIDATE DATA and apply the following cleaning rules:

    RULES:
    1. Normalize Skills: Consolidate duplicate or similar skills in 'hard_skills' and 'soft_skills' 
       (e.g., 'React', 'React.js', and 'ReactJS' should just be 'ReactJS').
    2. Flag Gaps: Look at the 'work_history'. If you detect an employment gap of 
       MORE THAN 3 MONTHS between two jobs, set 'is_gap_flagged' to true for the most recent job.
    3. Formatting: Ensure names, locations, and job titles are properly capitalized.
    4. DO NOT alter or remove the 'github_metrics', URLs, or contact info. Keep them exactly as is.

    CRITICAL INSTRUCTIONS:
    Return ONLY valid JSON matching this exact schema: {schema_instructions}

    RAW CANDIDATE DATA:
    {candidate_json}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1, 
            ),
        )
        
        # Validate the cleaned JSON back into our strict Pydantic model
        cleaned_candidate = CandidateSchema.model_validate_json(response.text)
        return cleaned_candidate
        
    except ValidationError as e:
        print(f"Agent 03 Validation Error: {e}")
        return candidate # If it fails, safely return the uncleaned original data
    except Exception as e:
        print(f"Agent 03 API Error: {e}")
        return candidate