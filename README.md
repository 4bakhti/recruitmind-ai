# RecruitMind AI 🧠
**Multi-Agent CV Intelligence System**

## 📁 Project Structure
```text
recruitmind/
├── README.md
├── .gitignore
├── .env.example            # Template for required API keys
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Project dependencies
├── core/
│   └── config.py           # Environment variables (API keys)
├── agents/
│   ├── agent_01_parser.py  # CV parsing & extraction (Gemini + PyMuPDF)
│   ├── agent_02_crawler.py # GitHub/LinkedIn crawling
│   ├── agent_03_fusion.py  # Data cleaning & fusion
│   ├── parser_agent.py     # CV parsing (Claude + pypdf/python-docx)
│   ├── github_crawler_agent.py  # GitHub profile enrichment (REST API)
│   ├── fusion_agent.py     # Pipeline owner: parse -> enrich -> reconcile
│   ├── job_fit_agent.py    # Candidate vs job scoring (local embeddings)
│   └── report_agent.py     # Report generation (Claude summary + Markdown)
├── models/
│   ├── candidate.py        # Pydantic schemas (CandidateSchema + Candidate)
│   ├── job_description.py  # JobDescription + JobFitResult
│   └── report.py           # CandidateReport + ScoreBreakdown
├── orchestrator/
│   ├── state.py            # PipelineState: the state flowing through the graph
│   ├── nodes.py            # One thin node per agent
│   └── graph.py            # LangGraph wiring + run_pipeline()
└── tests/
    └── test_*.py
```

## Pipeline

The Claude agents are wired together by a LangGraph state machine
(`orchestrator/graph.py`). Branches are edges, not buried `if`s:

```text
parse --+-- (parse failed) ---------------------------> END
        +-- (GitHub URL) --> crawl --> fuse --+
        +-- (no GitHub) ------------> fuse ---+
                                              |
                    +-- (no job) -------------+---> END
                    +-- (job) --> score --+-- (no scores) --> END
                                          +-- build_report --> END
```

Only parsing can end a run: every later stage degrades gracefully,
recording what went wrong in the state instead of raising. The parse node
retries once before giving up (`PARSE_MAX_ATTEMPTS` in
`orchestrator/nodes.py`).

```python
from orchestrator import run_pipeline

state = run_pipeline("resume.pdf", job)   # job is optional
state["candidate"]        # None only if parsing failed outright
state["report"]           # None if scoring/report was skipped or failed
state["steps_completed"]  # e.g. ["parse", "crawl", "fuse", "score", "report"]
state["errors"]           # why anything above is missing
```

`agents/fusion_agent.analyze_cv()` still works as the simple
straight-line path for callers that only want a fused Candidate.

## 🚀 Setup

Requires **Python 3.11** (the version the pinned dependencies are verified
against; 3.13 is too new for some pins, e.g. pydantic 2.6.1 and
PyMuPDF 1.23.21 have no 3.13 wheels).

Create and activate a project-local virtual environment, then install the
pinned dependencies:

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

All versions in requirements.txt are pinned exactly, so this reproduces
the verified environment. The first job-fit scoring run additionally
downloads the local embedding model (~90 MB) from Hugging Face into your
user cache.

Copy `.env.example` to `.env` and fill in your keys:

- `ANTHROPIC_API_KEY` — used by the Claude parser (`agents/parser_agent.py`)
- `GEMINI_API_KEY` — used by the Gemini pipeline (agents 01 & 03)
- `GITHUB_TOKEN` — optional, raises GitHub API rate limits for agent 02

## ▶️ Run

```bash
uvicorn main:app --reload
```

Then open http://localhost:8000/docs for the interactive API docs.

### Endpoints

| Method | Path                 | Description |
|--------|----------------------|-------------|
| GET    | `/health`            | Health check |
| POST   | `/parse-cv`          | Parse a single CV (PDF/DOCX) with the Claude parser agent; returns structured Candidate JSON |
| POST   | `/crawl-github`      | Fetch a GitHub profile summary for a username (query param `github_username`) |
| POST   | `/analyze-candidate` | Orchestrated pipeline: parse → GitHub enrich → fuse; returns the annotated Candidate |
| POST   | `/score-fit`         | `/analyze-candidate` + job-fit scoring against a job description |
| POST   | `/full-report`       | Everything: pipeline + scoring + recruiter report (JSON incl. Markdown in `report_markdown`) |
| POST   | `/run-pipeline`      | Orchestrated pipeline with its trace: returns candidate, scores, report, `steps_completed` and `errors` (job description optional) |
| POST   | `/api/v1/upload-cvs/`| Full Gemini pipeline: parse → crawl → fuse (bulk upload) |

Examples:

```bash
curl -X POST http://localhost:8000/parse-cv -F "file=@resume.pdf"

curl -X POST http://localhost:8000/score-fit \
  -F "file=@resume.pdf" \
  -F 'job_description={"title": "Backend Engineer", "required_skills": ["Python", "FastAPI"], "nice_to_have_skills": ["Kubernetes"], "description_text": "Build REST APIs."}'
```

(`job_description` is a JSON string in a form field because the request is
already multipart for the file upload.)

## 🧪 Tests

```bash
python -m unittest discover -s tests
```

- `tests/test_parser_agent.py` — Claude call is mocked, no API key needed
- `tests/test_github_crawler_agent.py` — hits the real GitHub API (skipped
  automatically if the unauthenticated rate limit is exhausted)
- `tests/test_fusion_agent.py` — fully offline (Claude and GitHub mocked)
- `tests/test_job_fit_agent.py` — scoring math is offline; the semantic
  tests load the local embedding model (first run downloads ~90 MB)
- `tests/test_report_agent.py` — Claude mocked; the endpoint test runs the
  whole pipeline in-process with real local embeddings
- `tests/test_orchestrator.py` — fully offline; every agent is mocked so the
  tests cover the graph's routing and failure handling, not the agents
