from typing import List, Literal

from pydantic import BaseModel, Field

from models.candidate import Candidate
from models.job_description import JobDescription, JobFitResult


class ScoreBreakdown(BaseModel):
    """Raw numbers behind the recommendation, including the weights, so
    the report is auditable without reading the code."""
    composite_score: float
    hard_skill_match_score: float
    semantic_fit_score: float
    hard_skill_weight: float
    semantic_weight: float


class CandidateReport(BaseModel):
    """Output of agents/report_agent.py.

    Only executive_summary is LLM-written; every other field is computed
    deterministically from the candidate and job-fit data.
    """
    executive_summary: str
    recommendation: Literal["Strong Fit", "Possible Fit", "Weak Fit"]
    top_strengths: List[str] = Field(default_factory=list)
    top_risks: List[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown

    # Full inputs nested for reference / auditability
    candidate: Candidate
    job: JobDescription
    job_fit: JobFitResult

    # Human-facing rendering of the sections above
    report_markdown: str = ""
