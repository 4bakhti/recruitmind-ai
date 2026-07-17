"""Fusion Agent.

Owns the full CV pipeline: parse (Claude) -> enrich (GitHub crawler) ->
reconcile. The reconcile step normalizes skill names, cross-references
GitHub languages against CV skills, flags employment gaps, and records
data-quality notes so downstream consumers know what is missing.

The pipeline never raises on partial failure: whatever data could be
gathered is returned, with the gaps described in data_quality_notes.
"""

import logging
import re
from datetime import date

from agents.github_crawler_agent import enrich_candidate
from agents.parser_agent import parse_cv
from models.candidate import Candidate, WorkEntry

logger = logging.getLogger(__name__)

GAP_THRESHOLD_DAYS = 90

# Static variant -> canonical mapping for common skill spellings.
# Deliberately simple; fuzzy matching is a future step if this proves
# insufficient. Keys must be lowercase. Shared with the job-fit agent.
SKILL_NORMALIZATION = {
    "react": "React", "reactjs": "React", "react.js": "React",
    "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
    "js": "JavaScript", "javascript": "JavaScript",
    "ts": "TypeScript", "typescript": "TypeScript",
    "py": "Python", "python": "Python", "python3": "Python",
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
    "k8s": "Kubernetes", "kubernetes": "Kubernetes",
    "golang": "Go", "go": "Go",
    "c#": "C#", "csharp": "C#", "c sharp": "C#",
    "c++": "C++", "cpp": "C++",
    ".net": ".NET", "dotnet": ".NET",
    "aws": "AWS", "amazon web services": "AWS",
    "gcp": "GCP", "google cloud": "GCP", "google cloud platform": "GCP",
    "html": "HTML", "html5": "HTML",
    "css": "CSS", "css3": "CSS",
    "vue": "Vue.js", "vuejs": "Vue.js", "vue.js": "Vue.js",
    "sklearn": "scikit-learn", "scikit-learn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch", "torch": "PyTorch",
    "ml": "Machine Learning", "machine learning": "Machine Learning",
    "docker": "Docker",
    "fastapi": "FastAPI",
    "django": "Django",
    "sql": "SQL",
    "mongo": "MongoDB", "mongodb": "MongoDB",
}


def normalize_skill(skill: str) -> str:
    """Map a single skill string to its canonical spelling."""
    cleaned = skill.strip()
    return SKILL_NORMALIZATION.get(cleaned.lower(), cleaned)


def normalize_skills(skills: list[str]) -> list[str]:
    """Normalize a skill list, deduplicating while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for skill in skills:
        canonical = normalize_skill(skill)
        if canonical and canonical.lower() not in seen:
            seen.add(canonical.lower())
            result.append(canonical)
    return result


# --- Date handling for gap detection ------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_PRESENT_WORDS = {"present", "current", "now", "ongoing", "today"}


def _parse_work_date(value: str | None) -> date | None:
    """Best-effort parse of the date strings CVs contain.

    Handles: "2020", "2020-01", "2020-01-15", "Jan 2020", "January 2020",
    "01/2020", and "Present"/"Current" (-> today). Returns None if the
    format is unrecognized.
    """
    if not value:
        return None
    v = value.strip().lower().rstrip(".")

    if v in _PRESENT_WORDS:
        return date.today()

    # ISO-ish: 2020 / 2020-01 / 2020-01-15 (also with slashes)
    m = re.fullmatch(r"(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?", v)
    if m:
        year = int(m.group(1))
        month = min(int(m.group(2) or 1), 12) or 1
        day = min(int(m.group(3) or 1), 28) or 1
        return date(year, month, day)

    # "Jan 2020" / "January 2020" (optional comma)
    m = re.fullmatch(r"([a-z]{3,9})\.?,?\s+(\d{4})", v)
    if m and m.group(1)[:3] in _MONTHS:
        return date(int(m.group(2)), _MONTHS[m.group(1)[:3]], 1)

    # "01/2020" / "1.2020"
    m = re.fullmatch(r"(\d{1,2})[/.](\d{4})", v)
    if m and 1 <= int(m.group(1)) <= 12:
        return date(int(m.group(2)), int(m.group(1)), 1)

    return None


def detect_employment_gaps(work_history: list[WorkEntry]) -> list[str]:
    """Return human-readable notes for gaps > GAP_THRESHOLD_DAYS between
    consecutive roles. Entries with unparseable dates are skipped."""
    dated = []
    for entry in work_history:
        start = _parse_work_date(entry.start_date)
        end = _parse_work_date(entry.end_date)
        if start is not None:
            dated.append((start, end, entry))
    dated.sort(key=lambda item: item[0])

    notes = []
    for (_, prev_end, prev), (next_start, _, nxt) in zip(dated, dated[1:]):
        if prev_end is None:
            continue
        gap_days = (next_start - prev_end).days
        if gap_days > GAP_THRESHOLD_DAYS:
            months = round(gap_days / 30)
            notes.append(
                f"Employment gap of ~{months} months between "
                f"{prev.company or 'previous role'} (ended {prev.end_date}) "
                f"and {nxt.company or 'next role'} "
                f"(started {nxt.start_date})"
            )
    return notes


# --- The fusion steps -----------------------------------------------------

def fuse(candidate: Candidate) -> Candidate:
    """Reconcile and annotate an already parsed (and possibly GitHub-
    enriched) candidate. Pure function of the candidate — no I/O — so it
    is easy to test. Safe to re-run: annotation fields are rebuilt from
    scratch each time."""
    candidate.skills = normalize_skills(candidate.skills)
    candidate.skills_from_github = []
    candidate.employment_gap_notes = detect_employment_gaps(
        candidate.work_history
    )
    candidate.data_quality_notes = []
    notes = candidate.data_quality_notes

    # Cross-reference GitHub languages against CV skills
    if candidate.github_profile is not None:
        cv_skills = {s.lower() for s in candidate.skills}
        for language in candidate.github_profile.top_languages:
            canonical = normalize_skill(language)
            if canonical.lower() not in cv_skills:
                candidate.skills_from_github.append(canonical)
                cv_skills.add(canonical.lower())
    else:
        notes.append("No GitHub data available")

    if not candidate.work_history:
        notes.append("No work history found in CV")
    elif len(candidate.work_history) == 1:
        notes.append("Work history has only 1 entry")

    undated = sum(
        1 for w in candidate.work_history
        if _parse_work_date(w.start_date) is None
    )
    if undated:
        notes.append(
            f"Could not parse dates for {undated} work history "
            f"entr{'y' if undated == 1 else 'ies'} — gap detection may be "
            f"incomplete"
        )

    if not candidate.skills:
        notes.append("No skills listed in CV")
    if not candidate.email:
        notes.append("No email found in CV")
    if not candidate.phone:
        notes.append("No phone number found in CV")
    if not candidate.education:
        notes.append("No education history found in CV")

    return candidate


def analyze_cv(file_path: str) -> Candidate | None:
    """Full pipeline: parse the CV, enrich with GitHub data, fuse.

    Returns None only if parsing produced nothing at all; every partial
    failure after that is annotated in data_quality_notes instead.
    """
    candidate = parse_cv(file_path)
    if candidate is None:
        logger.error("Fusion: parser returned no candidate for %s", file_path)
        return None

    # enrich_candidate never raises; on failure github_profile stays None
    candidate = enrich_candidate(candidate)

    return fuse(candidate)
