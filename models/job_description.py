from typing import List, Optional

from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    title: str
    company: Optional[str] = None
    required_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    min_years_experience: Optional[int] = None
    description_text: str = ""


class JobFitResult(BaseModel):
    """Output of agents/job_fit_agent.py.

    Every score comes with the lists that produced it, so a recruiter can
    see exactly why the number came out the way it did.
    """
    semantic_fit_score: float = Field(
        description="0-100; cosine similarity of candidate vs job embeddings"
    )
    hard_skill_match_score: float = Field(
        description="0-100; fraction of required_skills the candidate has"
    )
    composite_score: float = Field(
        description="Weighted average of the two scores above"
    )
    matched_required_skills: List[str] = Field(default_factory=list)
    gap_analysis: List[str] = Field(
        default_factory=list,
        description="required_skills the candidate is missing",
    )
    nice_to_have_matches: List[str] = Field(default_factory=list)
    scoring_explanation: str = ""
