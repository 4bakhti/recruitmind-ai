"""The RecruitMind pipeline as a LangGraph state machine.

    parse ──┬── (no candidate) ─────────────────────────► END
            ├── (github signal) ──► crawl ──► fuse ──┐
            └── (no github) ────────────────► fuse ──┤
                                                     │
                        ┌── (no job) ────────────────┴──► END
                        └── (job) ──► score ──┬── (no scores) ──► END
                                              └── build_report ──► END

Why a graph rather than the straight-line calls in fusion_agent.analyze_cv:

* The branches are explicit. Skipping the crawler when there is no GitHub
  URL, and stopping after fusion when no job was supplied, are edges you
  can see instead of `if` statements buried in a function.
* Every step's output lands in one inspectable state object, so a partial
  run (parsed but scoring failed) still returns everything it managed to
  produce, with the reasons in `errors`.
* Adding an agent means adding a node and an edge, not editing a call chain.

`analyze_cv` is deliberately left alone; it remains the simple path for
callers that only want a fused Candidate.
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from models.job_description import JobDescription
from orchestrator.nodes import (
    crawl_node,
    fuse_node,
    has_github_signal,
    parse_node,
    report_node,
    score_node,
)
from orchestrator.state import PipelineState, initial_state

logger = logging.getLogger(__name__)


# --- Conditional edges ----------------------------------------------------

def route_after_parse(state: PipelineState) -> str:
    """Dead-end on a failed parse; otherwise crawl only if there is a
    GitHub username to crawl."""
    if state.get("candidate") is None:
        return "end"
    return "crawl" if has_github_signal(state) else "fuse"


def route_after_fuse(state: PipelineState) -> str:
    """Scoring needs a job description; without one the fused candidate
    is the deliverable."""
    return "score" if state.get("job") is not None else "end"


def route_after_score(state: PipelineState) -> str:
    """No scores, no report — the report is built entirely from them."""
    return "report" if state.get("fit") is not None else "end"


# --- Graph construction ---------------------------------------------------

def build_graph():
    """Wire and compile the pipeline graph.

    Node names avoid the state keys they write ("build_report" writes
    "report"); LangGraph refuses to let a node shadow a channel.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("parse", parse_node)
    graph.add_node("crawl", crawl_node)
    graph.add_node("fuse", fuse_node)
    graph.add_node("score", score_node)
    graph.add_node("build_report", report_node)

    graph.set_entry_point("parse")

    graph.add_conditional_edges(
        "parse", route_after_parse,
        {"crawl": "crawl", "fuse": "fuse", "end": END},
    )
    graph.add_edge("crawl", "fuse")
    graph.add_conditional_edges(
        "fuse", route_after_fuse, {"score": "score", "end": END},
    )
    graph.add_conditional_edges(
        "score", route_after_score, {"report": "build_report", "end": END},
    )
    graph.add_edge("build_report", END)

    return graph.compile()


# Compiling is cheap and the graph is stateless, so one shared instance is
# enough; every run gets its own state dict.
_compiled = None


def get_graph():
    """Return the compiled graph, building it on first use."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


# --- Public entry point ---------------------------------------------------

def run_pipeline(
    file_path: str, job: Optional[JobDescription] = None
) -> PipelineState:
    """Run the full pipeline over one CV.

    With `job`, runs everything through report generation; without it,
    stops after fusion and returns the annotated Candidate.

    Never raises on pipeline failure. Inspect the result:
        state["candidate"]        None only if parsing failed outright
        state["fit"] / ["report"] None if that stage was skipped or failed
        state["errors"]           why anything is missing
        state["steps_completed"]  which nodes actually ran
    """
    result = get_graph().invoke(initial_state(file_path, job))
    logger.info(
        "Pipeline finished for %s — steps: %s, errors: %d",
        file_path, result.get("steps_completed"), len(result.get("errors", [])),
    )
    return result
