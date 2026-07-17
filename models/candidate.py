from datetime import date
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional

# Sub-Models (Building Blocks)

class Education(BaseModel):
    institution: str
    degree: str
    graduation_year: Optional[str] = None

class WorkExperience(BaseModel):
    company: str
    role: str
    start_date: str
    end_date: str
    description: Optional[str] = None
    # Agent 03 flags gaps in the timeline
    is_gap_flagged: bool = False 

class GithubMetrics(BaseModel):
    """Enriched data from Agent 02"""
    repos_count: int = 0
    total_commits: int = 0
    code_quality_score: Optional[float] = None

class MarketIntelligence(BaseModel):
    """Data from Agent 05"""
    salary_benchmark: Optional[str] = None
    talent_scarcity_score: Optional[int] = Field(None, description="1-100 score")



# Master Schema (The full unified profile)

class CandidateSchema(BaseModel):
    """
    The unified Candidate Profile that flows through the LangGraph Orchestrator.
    """
    # 1. Extracted by Agent 01 (CV Parser)
    full_name: str
    email: str 
    phone: Optional[str] = None
    location: Optional[str] = None
    
    education: List[Education] = Field(default_factory=list)
    work_history: List[WorkExperience] = Field(default_factory=list)
    
    # Skills are normalized by Agent 03 (e.g., React = ReactJS)
    hard_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    tools_and_certifications: List[str] = Field(default_factory=list)
    
    # 2. Enriched by Agent 02 (Link & Profile Crawler)
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    github_metrics: Optional[GithubMetrics] = None
    
    # 3. Compliance & Bias (Agent 04)
    # When true, name/gender/age signals are stripped for the final report
    is_anonymized: bool = False 
    
    # 4. Final Scoring (Agent 06 & 08)
    market_data: Optional[MarketIntelligence] = None
    job_fit_score: Optional[float] = Field(None, description="0-100 weighted match score")
    top_strengths: List[str] = Field(default_factory=list)
    top_risks: List[str] = Field(default_factory=list)


# Minimal parsed-candidate model used by the Claude-based parser
# (agents/parser_agent.py). Kept deliberately flexible: any field the
# parser can't find in the CV is left as None / empty.

class EducationEntry(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


class WorkEntry(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class NotableRepo(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    stars: int = 0
    language: Optional[str] = None
    last_updated: Optional[str] = None  # ISO timestamp of the last push


class GithubProfile(BaseModel):
    """Enrichment data from agents/github_crawler_agent.py."""
    public_repos: int = 0
    top_languages: List[str] = Field(default_factory=list)
    notable_repos: List[NotableRepo] = Field(default_factory=list)
    total_stars: int = 0
    account_created: Optional[date] = None
    contribution_summary: Optional[str] = None


class Candidate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    education: List[EducationEntry] = Field(default_factory=list)
    work_history: List[WorkEntry] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    github_username: Optional[str] = None
    github_profile: Optional[GithubProfile] = None

    # Annotations added by the Fusion Agent (agents/fusion_agent.py)
    skills_from_github: List[str] = Field(default_factory=list)
    employment_gap_notes: List[str] = Field(default_factory=list)
    data_quality_notes: List[str] = Field(default_factory=list)