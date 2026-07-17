"""Tests for agents/parser_agent.py.

The Claude API call is mocked — no API key or network needed.
Run from the project root:  python -m unittest tests.test_parser_agent
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents import parser_agent
from models.candidate import Candidate

SAMPLE_CV_TEXT = """
Jane Doe
Email: jane.doe@example.com | Phone: +1 555 010 2030 | Berlin, Germany

EDUCATION
B.Sc. Computer Science, Technical University of Munich, 2019

EXPERIENCE
Backend Engineer, Acme GmbH (2020-01 - 2023-06)
Built REST APIs with Python and FastAPI.

SKILLS
Python, FastAPI, PostgreSQL, Docker
"""

CLAUDE_JSON_RESPONSE = json.dumps(
    {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": "+1 555 010 2030",
        "location": "Berlin, Germany",
        "education": [
            {
                "degree": "B.Sc. Computer Science",
                "institution": "Technical University of Munich",
                "year": "2019",
            }
        ],
        "work_history": [
            {
                "company": "Acme GmbH",
                "title": "Backend Engineer",
                "start_date": "2020-01",
                "end_date": "2023-06",
                "description": "Built REST APIs with Python and FastAPI.",
            }
        ],
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "raw_text": None,
    }
)


def _fake_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class ParseCvTextTests(unittest.TestCase):
    def test_parses_valid_json_into_candidate(self):
        with patch.object(parser_agent, "_get_client") as get_client:
            get_client.return_value.messages.create.return_value = _fake_response(
                CLAUDE_JSON_RESPONSE
            )
            candidate = parser_agent.parse_cv_text(SAMPLE_CV_TEXT)

        self.assertIsInstance(candidate, Candidate)
        self.assertEqual(candidate.name, "Jane Doe")
        self.assertEqual(candidate.email, "jane.doe@example.com")
        self.assertEqual(len(candidate.education), 1)
        self.assertEqual(candidate.education[0].year, "2019")
        self.assertEqual(candidate.work_history[0].company, "Acme GmbH")
        self.assertIn("Python", candidate.skills)
        self.assertEqual(candidate.raw_text, SAMPLE_CV_TEXT)

    def test_tolerates_markdown_fences(self):
        fenced = f"```json\n{CLAUDE_JSON_RESPONSE}\n```"
        with patch.object(parser_agent, "_get_client") as get_client:
            get_client.return_value.messages.create.return_value = _fake_response(fenced)
            candidate = parser_agent.parse_cv_text(SAMPLE_CV_TEXT)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.name, "Jane Doe")

    def test_returns_none_on_invalid_json(self):
        with patch.object(parser_agent, "_get_client") as get_client:
            get_client.return_value.messages.create.return_value = _fake_response(
                "Sorry, I cannot parse this CV."
            )
            candidate = parser_agent.parse_cv_text(SAMPLE_CV_TEXT)

        self.assertIsNone(candidate)

    def test_returns_none_on_empty_input(self):
        self.assertIsNone(parser_agent.parse_cv_text("   "))


class ExtractTextTests(unittest.TestCase):
    def test_unsupported_extension_returns_empty(self):
        self.assertEqual(parser_agent.extract_text("cv.txt"), "")

    def test_missing_file_returns_empty(self):
        self.assertEqual(parser_agent.extract_text("does_not_exist.pdf"), "")


if __name__ == "__main__":
    unittest.main()
