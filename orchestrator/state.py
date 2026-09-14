"""Shared state for the LangGraph pipeline.

One TypedDict flows through every node. Nodes return *partial* dicts;
LangGraph merges them into the running state. The two `Annotated[...,
operator.add]` fields are accumulators — each node appends to them
instead of overwriting, so the finished state carries a full trace of
what ran and what went wrong.
"""

import operator
from typing import Annotated, Optional, TypedDict

from models.candidate import Candidate
from models.job_description import JobDescription, JobFitResult
from models.report import CandidateReport


class PipelineState(TypedDict, total=False):
    """State threaded through the RecruitMind graph.

    Inputs:
        file_path: CV to analyze (PDF or DOCX).
        job:       target role; when None the graph stops after fusion.

    Produced by the nodes:
        candidate: parsed + enriched + annotated Candidate.
        fit:       job-fit scores (only when `job` was supplied).
        report:    recruiter report (only when scoring succeeded).

    Accumulated across nodes:
        errors:          human-readable failures, in the order they happened.
        steps_completed: node names that ran to completion, in order.
    """

    file_path: str
    job: Optional[JobDescription]

    candidate: Optional[Candidate]
    fit: Optional[JobFitResult]
    report: Optional[CandidateReport]

    errors: Annotated[list[str], operator.add]
    steps_completed: Annotated[list[str], operator.add]


def initial_state(
    file_path: str, job: Optional[JobDescription] = None
) -> PipelineState:
    """Build a fully-populated starting state.

    Every key is set explicitly: LangGraph rejects updates naming keys the
    schema doesn't declare, and starting from a complete dict keeps the
    node signatures honest about what they can read.
    """
    return {
        "file_path": file_path,
        "job": job,
        "candidate": None,
        "fit": None,
        "report": None,
        "errors": [],
        "steps_completed": [],
    }
