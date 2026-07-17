"""Report Generator Agent.

Turns a fused Candidate + JobFitResult into a CandidateReport. The split
is strict and deliberate:

- Claude writes ONLY the executive summary prose.
- Recommendation, strengths, risks, and the score breakdown are computed
  deterministically from the data, so reports stay consistent and
  auditable — the same inputs always produce the same assessment.

Rendering is Markdown for now; PDF export is a possible future step.
"""

import json
import logging

import anthropic

from agents.job_fit_agent import HARD_SKILL_WEIGHT, SEMANTIC_WEIGHT
from models.candidate import Candidate
from models.job_description import JobDescription, JobFitResult
from models.report import CandidateReport, ScoreBreakdown

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

# Recommendation bands — tune here, nowhere else
STRONG_FIT_THRESHOLD = 75.0
POSSIBLE_FIT_THRESHOLD = 50.0

STRONG_FIT = "Strong Fit"
POSSIBLE_FIT = "Possible Fit"
WEAK_FIT = "Weak Fit"

# Deterministic selection limits and cutoffs
TOP_ITEMS_LIMIT = 3
NOTABLE_STARS_THRESHOLD = 100     # a single repo this starred is a signal
STRONG_GITHUB_STARS_THRESHOLD = 50
LOW_HARD_SKILL_THRESHOLD = 50.0
LOW_SEMANTIC_THRESHOLD = 40.0

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    # Lazy so importing this module never requires an API key
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def recommendation_from_score(composite_score: float) -> str:
    """Deterministic banding of the composite score. Never LLM-decided."""
    if composite_score >= STRONG_FIT_THRESHOLD:
        return STRONG_FIT
    if composite_score >= POSSIBLE_FIT_THRESHOLD:
        return POSSIBLE_FIT
    return WEAK_FIT


def select_top_strengths(
    candidate: Candidate, fit: JobFitResult
) -> list[str]:
    """Rule-based strength picks, strongest signal first."""
    strengths: list[str] = []

    if fit.matched_required_skills:
        total = len(fit.matched_required_skills) + len(fit.gap_analysis)
        strengths.append(
            f"Matches {len(fit.matched_required_skills)}/{total} required "
            f"skills: {', '.join(fit.matched_required_skills)}"
        )

    if fit.nice_to_have_matches:
        strengths.append(
            "Nice-to-have skills present: "
            + ", ".join(fit.nice_to_have_matches)
        )

    gh = candidate.github_profile
    if gh is not None:
        if gh.notable_repos and gh.notable_repos[0].stars >= NOTABLE_STARS_THRESHOLD:
            top = gh.notable_repos[0]
            strengths.append(
                f"Notable open-source project: {top.name} "
                f"({top.stars} stars{f', {top.language}' if top.language else ''})"
            )
        if gh.total_stars >= STRONG_GITHUB_STARS_THRESHOLD:
            strengths.append(
                f"Strong GitHub presence: {gh.total_stars} stars across "
                f"{gh.public_repos} public repos"
            )
        if gh.contribution_summary and gh.contribution_summary.lower().startswith("active"):
            strengths.append(f"GitHub activity: {gh.contribution_summary}")

    # Work history that demonstrates a matched required skill in practice
    for entry in candidate.work_history:
        text = f"{entry.title or ''} {entry.description or ''}".lower()
        demonstrated = [
            s for s in fit.matched_required_skills if s.lower() in text
        ]
        if demonstrated:
            where = f" at {entry.company}" if entry.company else ""
            strengths.append(
                f"Hands-on experience with "
                f"{', '.join(demonstrated)}{where}"
            )
            break  # one work-history strength is enough

    return strengths[:TOP_ITEMS_LIMIT]


def select_top_risks(candidate: Candidate, fit: JobFitResult) -> list[str]:
    """Rule-based risk picks, most damaging first."""
    risks: list[str] = []

    if fit.gap_analysis:
        risks.append(
            "Missing required skills: " + ", ".join(fit.gap_analysis)
        )
    if fit.hard_skill_match_score < LOW_HARD_SKILL_THRESHOLD:
        risks.append(
            f"Covers less than half of the required skills "
            f"(hard-skill score {fit.hard_skill_match_score}/100)"
        )

    risks.extend(candidate.employment_gap_notes)

    if fit.semantic_fit_score < LOW_SEMANTIC_THRESHOLD:
        risks.append(
            f"CV content has low semantic overlap with the job description "
            f"({fit.semantic_fit_score}/100)"
        )

    risks.extend(candidate.data_quality_notes)

    return risks[:TOP_ITEMS_LIMIT]


def _build_summary_prompt(
    candidate: Candidate,
    job: JobDescription,
    fit: JobFitResult,
    recommendation: str,
    strengths: list[str],
    risks: list[str],
) -> str:
    facts = {
        "candidate_name": candidate.name,
        "candidate_location": candidate.location,
        "job_title": job.title,
        "company": job.company,
        "recommendation": recommendation,
        "composite_score": fit.composite_score,
        "matched_required_skills": fit.matched_required_skills,
        "missing_required_skills": fit.gap_analysis,
        "top_strengths": strengths,
        "top_risks": risks,
        "most_recent_role": (
            candidate.work_history[0].model_dump(exclude={"description"})
            if candidate.work_history else None
        ),
    }
    return (
        "You are writing the executive summary of a recruiting report.\n"
        "Using ONLY the facts below, write 3-4 sentences covering: who the "
        "candidate is, their strongest match areas for this role, the main "
        "concerns, and a final clear recommendation line that MUST agree "
        "with the given recommendation value (do not soften or change it).\n"
        "Return plain prose only — no markdown, no bullet points, no "
        "preamble, no invented facts.\n\n"
        f"FACTS:\n{json.dumps(facts, ensure_ascii=False)}"
    )


def generate_executive_summary(
    candidate: Candidate,
    job: JobDescription,
    fit: JobFitResult,
    recommendation: str,
    strengths: list[str],
    risks: list[str],
) -> str:
    """The one LLM-written piece of the report. Falls back to a plain
    deterministic sentence if the API call fails — the report must never
    be lost over summary prose."""
    prompt = _build_summary_prompt(
        candidate, job, fit, recommendation, strengths, risks
    )
    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return next(
            b.text for b in response.content if b.type == "text"
        ).strip()
    except Exception as e:
        logger.error("Executive summary generation failed: %s", e)
        return (
            f"{candidate.name or 'The candidate'} scored "
            f"{fit.composite_score}/100 overall for the {job.title} role "
            f"({recommendation}). Automatic summary generation failed; see "
            f"the strengths, risks, and score breakdown below."
        )


def generate_report(
    candidate: Candidate, job: JobDescription, fit: JobFitResult
) -> CandidateReport:
    """Assemble the full report. Everything except executive_summary is
    deterministic."""
    recommendation = recommendation_from_score(fit.composite_score)
    strengths = select_top_strengths(candidate, fit)
    risks = select_top_risks(candidate, fit)

    report = CandidateReport(
        executive_summary=generate_executive_summary(
            candidate, job, fit, recommendation, strengths, risks
        ),
        recommendation=recommendation,
        top_strengths=strengths,
        top_risks=risks,
        score_breakdown=ScoreBreakdown(
            composite_score=fit.composite_score,
            hard_skill_match_score=fit.hard_skill_match_score,
            semantic_fit_score=fit.semantic_fit_score,
            hard_skill_weight=HARD_SKILL_WEIGHT,
            semantic_weight=SEMANTIC_WEIGHT,
        ),
        candidate=candidate,
        job=job,
        job_fit=fit,
    )
    report.report_markdown = render_report_markdown(report)
    return report


def render_report_markdown(report: CandidateReport) -> str:
    """Render the human-facing Markdown document.

    PDF export is a possible future step; Markdown is the deliverable
    for now.
    """
    sb = report.score_breakdown
    name = report.candidate.name or "Unknown Candidate"
    job_line = report.job.title + (
        f" — {report.job.company}" if report.job.company else ""
    )

    lines = [
        f"# Candidate Report: {name}",
        "",
        f"**Position:** {job_line}",
        f"**Recommendation:** {report.recommendation} "
        f"(composite score {sb.composite_score}/100)",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "## Top Strengths",
        "",
    ]
    if report.top_strengths:
        lines += [f"{i}. {s}" for i, s in enumerate(report.top_strengths, 1)]
    else:
        lines.append("_No notable strengths identified._")

    lines += ["", "## Top Risks", ""]
    if report.top_risks:
        lines += [f"{i}. {r}" for i, r in enumerate(report.top_risks, 1)]
    else:
        lines.append("_No notable risks identified._")

    lines += [
        "",
        "## Score Breakdown",
        "",
        "| Metric | Score | Weight |",
        "|--------|-------|--------|",
        f"| Hard skill match | {sb.hard_skill_match_score} | "
        f"{sb.hard_skill_weight:.0%} |",
        f"| Semantic fit | {sb.semantic_fit_score} | "
        f"{sb.semantic_weight:.0%} |",
        f"| **Composite** | **{sb.composite_score}** | — |",
        "",
    ]
    if report.job_fit.matched_required_skills:
        lines.append(
            "Matched required skills: "
            + ", ".join(report.job_fit.matched_required_skills)
        )
    if report.job_fit.gap_analysis:
        lines.append(
            "Missing required skills: "
            + ", ".join(report.job_fit.gap_analysis)
        )
    if report.job_fit.scoring_explanation:
        lines += ["", f"_{report.job_fit.scoring_explanation}_"]

    return "\n".join(lines) + "\n"
