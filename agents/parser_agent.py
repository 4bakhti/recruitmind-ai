"""CV Parser Agent (Claude).

Extracts raw text from a PDF or DOCX file, then asks Claude to structure it
into the minimal `Candidate` model. Runs standalone — it only needs
ANTHROPIC_API_KEY, not the Gemini/GitHub keys required by core.config.
"""

import json
import logging
from pathlib import Path

import anthropic
from docx import Document
from dotenv import load_dotenv
from pydantic import ValidationError
from pypdf import PdfReader

from models.candidate import Candidate

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    # Lazy so importing this module (e.g. in tests) never requires an API key
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def extract_text(file_path: str) -> str:
    """Extract raw text from a PDF or DOCX file. Returns "" on failure."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".docx":
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        logger.error("Unsupported file type: %s", suffix)
        return ""
    except Exception:
        logger.exception("Failed to extract text from %s", file_path)
        return ""


def _build_prompt(raw_text: str) -> str:
    schema = json.dumps(Candidate.model_json_schema())
    return (
        "You are a CV parsing assistant. Extract structured data from the CV "
        "text below.\n\n"
        "Return ONLY a valid JSON object matching this JSON schema — no "
        "markdown fences, no preamble, no explanation:\n"
        f"{schema}\n\n"
        "Rules:\n"
        "- If a field is not present in the CV, use null (or [] for lists).\n"
        "- Do not invent information that is not in the CV.\n"
        "- Leave raw_text as null; it is filled in separately.\n\n"
        f"CV TEXT:\n{raw_text}"
    )


def parse_cv_text(raw_text: str) -> Candidate | None:
    """Send raw CV text to Claude and parse the response into a Candidate.

    Returns None on any failure (empty input, API error, bad JSON) — never
    raises.
    """
    if not raw_text.strip():
        logger.warning("No text provided to parse")
        return None

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": _build_prompt(raw_text)}],
        )
        text = next(b.text for b in response.content if b.type == "text").strip()

        # Defensive: tolerate markdown fences despite the prompt forbidding them
        if text.startswith("```"):
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in model response")
            text = text[start : end + 1]

        candidate = Candidate.model_validate_json(text)
        candidate.raw_text = raw_text
        return candidate

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.error("Failed to parse Claude response into Candidate: %s", e)
        return None
    except anthropic.APIError as e:
        logger.error("Claude API error: %s", e)
        return None
    except Exception:
        logger.exception("Unexpected error while parsing CV")
        return None


def parse_cv(file_path: str) -> Candidate | None:
    """Full pipeline: file path -> raw text -> structured Candidate."""
    raw_text = extract_text(file_path)
    if not raw_text.strip():
        logger.error("No text could be extracted from %s", file_path)
        return None
    return parse_cv_text(raw_text)
