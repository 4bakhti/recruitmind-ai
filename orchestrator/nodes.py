"""Graph nodes.

Each node is a thin wrapper over an existing agent: it reads what it
needs from the state and returns only the keys it changed. No analysis
logic lives here — that stays in agents/ — so the graph stays a wiring
diagram you can read in one sitting.

Nodes never raise. A failure becomes an entry in `errors` and a None
result, and the conditional edges in graph.py decide where to go from
there.
"""

import logging
import time

from agents.fusion_agent import fuse
from agents.github_crawler_agent import enrich_candidate, extract_github_username
from agents.job_fit_agent import score_fit
from agents.parser_agent import parse_cv
from agents.report_agent import generate_report
from orchestrator.state import PipelineState

logger = logging.getLogger(__name__)

# The parser is the one node whose failure kills the whole run, and its
# failure modes (API blip, truncated response) are often transient — so
# it is the one node worth retrying. Module-level so tests can shrink them.
PARSE_MAX_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 2.0


def has_github_signal(state: PipelineState) -> bool:
    """Whether the crawler has a username to work with.

    Mirrors the lookup enrich_candidate does internally, so the graph can
    skip the node entirely rather than calling it to do nothing.
    """
    candidate = state.get("candidate")
    if candidate is None:
        return False
    if candidate.github_username:
        return True
    return bool(extract_github_username(candidate.raw_text or ""))


def parse_node(state: PipelineState) -> PipelineState:
    """Extract text from the CV and structure it with Claude."""
    file_path = state["file_path"]

    for attempt in range(1, PARSE_MAX_ATTEMPTS + 1):
        candidate = parse_cv(file_path)
        if candidate is not None:
            return {"candidate": candidate, "steps_completed": ["parse"]}
        if attempt < PARSE_MAX_ATTEMPTS:
            logger.warning(
                "Parse attempt %d/%d produced nothing for %s — retrying",
                attempt, PARSE_MAX_ATTEMPTS, file_path,
            )
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.error("Parsing failed after %d attempts: %s",
                 PARSE_MAX_ATTEMPTS, file_path)
    return {
        "candidate": None,
        "errors": [
            f"Could not parse the CV after {PARSE_MAX_ATTEMPTS} attempt(s). "
            "The file may be an image-only PDF, or the parser API call failed."
        ],
    }


def crawl_node(state: PipelineState) -> PipelineState:
    """Enrich the candidate with their public GitHub profile."""
    candidate = state["candidate"]
    enriched = enrich_candidate(candidate)

    # enrich_candidate swallows its own failures, so an absent profile
    # here means the fetch did not work — worth saying out loud, but not
    # worth stopping for.
    if enriched.github_profile is None:
        return {
            "candidate": enriched,
            "errors": [
                "GitHub enrichment found a username but could not fetch the "
                "profile (unknown user, rate limit, or network error)."
            ],
        }
    return {"candidate": enriched, "steps_completed": ["crawl"]}


def fuse_node(state: PipelineState) -> PipelineState:
    """Normalize skills, cross-reference GitHub, flag gaps and data holes."""
    return {
        "candidate": fuse(state["candidate"]),
        "steps_completed": ["fuse"],
    }


def score_node(state: PipelineState) -> PipelineState:
    """Score the fused candidate against the job description."""
    try:
        fit = score_fit(state["candidate"], state["job"])
    except Exception as exc:  # embedding model load / runtime failure
        logger.exception("Job-fit scoring failed")
        return {"fit": None, "errors": [f"Job-fit scoring failed: {exc}"]}
    return {"fit": fit, "steps_completed": ["score"]}


def report_node(state: PipelineState) -> PipelineState:
    """Assemble the recruiter-facing report."""
    try:
        report = generate_report(state["candidate"], state["job"], state["fit"])
    except Exception as exc:
        logger.exception("Report generation failed")
        return {"report": None, "errors": [f"Report generation failed: {exc}"]}
    return {"report": report, "steps_completed": ["report"]}
