"""Job-Fit Scoring Agent.

Compares a fused Candidate against a JobDescription and produces a
transparent JobFitResult: a semantic similarity score (local
sentence-transformers embeddings), an exact hard-skill match score, and
the skill lists explaining both.

Scoring is deliberately debuggable: the result carries the matched and
missing skill lists plus a plain-text explanation of the composite math.

Note: for a single candidate-vs-job pair we embed and compare directly —
no vector store. If we later need to search MANY stored candidates
against a job (or vice versa), that is the point to introduce ChromaDB
as a persistent index; do not add it before then.
"""

import logging

from agents.fusion_agent import normalize_skill
from models.candidate import Candidate
from models.job_description import JobDescription, JobFitResult

logger = logging.getLogger(__name__)

# Composite weighting — tune here, nowhere else
HARD_SKILL_WEIGHT = 0.6
SEMANTIC_WEIGHT = 0.4

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs locally

_model = None


def _get_model():
    # Lazy: importing this module must not load torch / download weights.
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def candidate_to_text(candidate: Candidate) -> str:
    """Flatten the candidate's skills and work history into one blob for
    embedding."""
    parts: list[str] = []
    if candidate.skills:
        parts.append("Skills: " + ", ".join(candidate.skills))
    if candidate.skills_from_github:
        parts.append(
            "Skills seen on GitHub: " + ", ".join(candidate.skills_from_github)
        )
    for entry in candidate.work_history:
        line = " ".join(
            filter(None, [entry.title, "at" if entry.company else None,
                          entry.company, "—", entry.description])
        )
        parts.append(line)
    return "\n".join(parts)


def job_to_text(job: JobDescription) -> str:
    parts = [job.title]
    if job.required_skills:
        parts.append("Required skills: " + ", ".join(job.required_skills))
    if job.nice_to_have_skills:
        parts.append("Nice to have: " + ", ".join(job.nice_to_have_skills))
    if job.description_text:
        parts.append(job.description_text)
    return "\n".join(parts)


def semantic_fit_score(candidate: Candidate, job: JobDescription) -> float:
    """Cosine similarity of the two embeddings, mapped to 0-100.

    Negative similarity (essentially unrelated texts) clamps to 0.
    """
    candidate_text = candidate_to_text(candidate)
    job_text = job_to_text(job)
    if not candidate_text.strip() or not job_text.strip():
        logger.warning("Empty text for embedding; semantic score = 0")
        return 0.0

    model = _get_model()
    embeddings = model.encode([candidate_text, job_text])
    # model.similarity is sentence-transformers' built-in cosine similarity
    cosine = float(model.similarity(embeddings[0], embeddings[1]).item())
    return round(max(0.0, cosine) * 100, 1)


def hard_skill_match(
    candidate: Candidate, job: JobDescription
) -> tuple[float, list[str], list[str]]:
    """Exact (post-normalization, case-insensitive) matching of
    required_skills against the candidate's skills + skills_from_github.

    Returns (score 0-100, matched_required_skills, gap_analysis).
    """
    candidate_skills = {
        normalize_skill(s).lower()
        for s in candidate.skills + candidate.skills_from_github
    }

    matched: list[str] = []
    missing: list[str] = []
    for required in job.required_skills:
        canonical = normalize_skill(required)
        if canonical.lower() in candidate_skills:
            matched.append(canonical)
        else:
            missing.append(canonical)

    if not job.required_skills:
        # Vacuously satisfied — nothing was required
        return 100.0, [], []

    score = round(100 * len(matched) / len(job.required_skills), 1)
    return score, matched, missing


def nice_to_have_matches(
    candidate: Candidate, job: JobDescription
) -> list[str]:
    candidate_skills = {
        normalize_skill(s).lower()
        for s in candidate.skills + candidate.skills_from_github
    }
    return [
        normalize_skill(s)
        for s in job.nice_to_have_skills
        if normalize_skill(s).lower() in candidate_skills
    ]


def composite_score(hard_score: float, semantic_score: float) -> float:
    return round(
        HARD_SKILL_WEIGHT * hard_score + SEMANTIC_WEIGHT * semantic_score, 1
    )


def score_fit(candidate: Candidate, job: JobDescription) -> JobFitResult:
    """Score a fused candidate against a job description."""
    hard_score, matched, missing = hard_skill_match(candidate, job)
    semantic = semantic_fit_score(candidate, job)
    composite = composite_score(hard_score, semantic)

    explanation = (
        f"composite {composite} = "
        f"{int(HARD_SKILL_WEIGHT * 100)}% x hard-skill {hard_score} "
        f"({len(matched)}/{len(job.required_skills)} required skills matched) "
        f"+ {int(SEMANTIC_WEIGHT * 100)}% x semantic {semantic} "
        f"(embedding cosine similarity, {EMBEDDING_MODEL})"
    )

    return JobFitResult(
        semantic_fit_score=semantic,
        hard_skill_match_score=hard_score,
        composite_score=composite,
        matched_required_skills=matched,
        gap_analysis=missing,
        nice_to_have_matches=nice_to_have_matches(candidate, job),
        scoring_explanation=explanation,
    )
