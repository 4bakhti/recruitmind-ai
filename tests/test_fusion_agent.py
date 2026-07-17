"""Tests for agents/fusion_agent.py.

Fully offline: the Claude call is mocked (same pattern as the parser
tests) and the GitHub fetch is mocked with a fixed profile.
Run from the project root:  python -m unittest tests.test_fusion_agent
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from docx import Document

from agents import fusion_agent
from agents.fusion_agent import (
    detect_employment_gaps,
    fuse,
    normalize_skill,
    normalize_skills,
)
from models.candidate import Candidate, GithubProfile, WorkEntry

SAMPLE_CV_TEXT = """
Jane Doe
Email: jane.doe@example.com | Berlin, Germany
GitHub: https://github.com/janedoe

EXPERIENCE
Backend Engineer, Acme GmbH (2020-01 - 2020-06)
Built REST APIs with Python and FastAPI.

Senior Engineer, Beta AG (2021-03 - Present)
Platform work.

SKILLS
python, ReactJS, postgres
"""

CLAUDE_JSON_RESPONSE = json.dumps(
    {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": None,
        "location": "Berlin, Germany",
        "education": [],
        "work_history": [
            {
                "company": "Acme GmbH",
                "title": "Backend Engineer",
                "start_date": "2020-01",
                "end_date": "2020-06",
                "description": "Built REST APIs with Python and FastAPI.",
            },
            {
                "company": "Beta AG",
                "title": "Senior Engineer",
                "start_date": "2021-03",
                "end_date": "Present",
                "description": "Platform work.",
            },
        ],
        "skills": ["python", "ReactJS", "postgres"],
        "raw_text": None,
    }
)

FAKE_GITHUB_PROFILE = GithubProfile(
    public_repos=12,
    top_languages=["Python", "Rust", "TypeScript"],
    total_stars=40,
    contribution_summary="Active — 5 of 12 public repos pushed to in the last year",
)


class SkillNormalizationTests(unittest.TestCase):
    def test_variants_map_to_canonical(self):
        self.assertEqual(normalize_skill("ReactJS"), "React")
        self.assertEqual(normalize_skill("react.js"), "React")
        self.assertEqual(normalize_skill("  python "), "Python")
        self.assertEqual(normalize_skill("postgres"), "PostgreSQL")
        self.assertEqual(normalize_skill("k8s"), "Kubernetes")

    def test_unknown_skill_kept_verbatim(self):
        self.assertEqual(normalize_skill("Terraform"), "Terraform")

    def test_normalize_list_dedupes_preserving_order(self):
        self.assertEqual(
            normalize_skills(["ReactJS", "React", "python", "Python3", "AWS"]),
            ["React", "Python", "AWS"],
        )


class EmploymentGapTests(unittest.TestCase):
    def test_clear_gap_is_flagged(self):
        history = [
            WorkEntry(company="Acme", start_date="2020-01", end_date="2020-06"),
            WorkEntry(company="Beta", start_date="2021-03", end_date="Present"),
        ]
        notes = detect_employment_gaps(history)
        self.assertEqual(len(notes), 1)
        self.assertIn("Acme", notes[0])
        self.assertIn("Beta", notes[0])
        self.assertIn("~9 months", notes[0])

    def test_back_to_back_roles_not_flagged(self):
        history = [
            WorkEntry(company="Acme", start_date="2020-01", end_date="2021-02"),
            WorkEntry(company="Beta", start_date="2021-03", end_date="Present"),
        ]
        self.assertEqual(detect_employment_gaps(history), [])

    def test_unordered_input_still_detected(self):
        history = [
            WorkEntry(company="Beta", start_date="Jan 2022", end_date="Present"),
            WorkEntry(company="Acme", start_date="2020-01", end_date="2020-06"),
        ]
        notes = detect_employment_gaps(history)
        self.assertEqual(len(notes), 1)

    def test_unparseable_dates_skipped(self):
        history = [
            WorkEntry(company="Acme", start_date="a while ago", end_date="later"),
            WorkEntry(company="Beta", start_date="2021-03", end_date="Present"),
        ]
        self.assertEqual(detect_employment_gaps(history), [])


class FuseTests(unittest.TestCase):
    def test_github_languages_added_as_inferred_skills(self):
        candidate = Candidate(
            skills=["python", "ReactJS"],
            github_profile=FAKE_GITHUB_PROFILE.model_copy(deep=True),
        )
        fused = fuse(candidate)
        self.assertEqual(fused.skills, ["Python", "React"])
        # Python already in CV skills -> only the genuinely new ones
        self.assertEqual(fused.skills_from_github, ["Rust", "TypeScript"])

    def test_missing_data_is_noted_not_fatal(self):
        fused = fuse(Candidate(name="Empty CV"))
        self.assertIn("No GitHub data available", fused.data_quality_notes)
        self.assertIn("No work history found in CV", fused.data_quality_notes)
        self.assertIn("No skills listed in CV", fused.data_quality_notes)
        self.assertIn("No email found in CV", fused.data_quality_notes)

    def test_fuse_is_idempotent(self):
        candidate = Candidate(
            skills=["python"],
            github_profile=FAKE_GITHUB_PROFILE.model_copy(deep=True),
        )
        once = fuse(candidate)
        twice = fuse(once)
        self.assertEqual(twice.skills_from_github, ["Rust", "TypeScript"])
        self.assertEqual(
            once.data_quality_notes.count("No email found in CV"), 1
        )


def _fake_claude_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class EndToEndTests(unittest.TestCase):
    def _write_sample_docx(self, directory: str) -> str:
        path = Path(directory) / "sample_cv.docx"
        doc = Document()
        for line in SAMPLE_CV_TEXT.strip().splitlines():
            doc.add_paragraph(line)
        doc.save(str(path))
        return str(path)

    def test_analyze_cv_full_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            cv_path = self._write_sample_docx(tmp)

            with patch("agents.parser_agent._get_client") as get_client, \
                 patch(
                     "agents.github_crawler_agent.fetch_github_profile",
                     return_value=FAKE_GITHUB_PROFILE.model_copy(deep=True),
                 ) as fetch:
                get_client.return_value.messages.create.return_value = (
                    _fake_claude_response(CLAUDE_JSON_RESPONSE)
                )
                candidate = fusion_agent.analyze_cv(cv_path)

        self.assertIsNotNone(candidate)
        fetch.assert_called_once_with("janedoe")  # extracted from raw_text

        # Skills normalized, GitHub-only languages tagged separately
        self.assertEqual(candidate.skills, ["Python", "React", "PostgreSQL"])
        self.assertEqual(candidate.skills_from_github, ["Rust", "TypeScript"])

        # The 2020-06 -> 2021-03 gap is flagged
        self.assertEqual(len(candidate.employment_gap_notes), 1)
        self.assertIn("Acme GmbH", candidate.employment_gap_notes[0])

        # Missing phone/education noted; GitHub note absent since crawl worked
        self.assertIn("No phone number found in CV", candidate.data_quality_notes)
        self.assertIn(
            "No education history found in CV", candidate.data_quality_notes
        )
        self.assertNotIn("No GitHub data available", candidate.data_quality_notes)

    def test_analyze_cv_survives_github_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cv_path = self._write_sample_docx(tmp)

            with patch("agents.parser_agent._get_client") as get_client, \
                 patch(
                     "agents.github_crawler_agent.fetch_github_profile",
                     return_value=None,
                 ):
                get_client.return_value.messages.create.return_value = (
                    _fake_claude_response(CLAUDE_JSON_RESPONSE)
                )
                candidate = fusion_agent.analyze_cv(cv_path)

        self.assertIsNotNone(candidate)
        self.assertIsNone(candidate.github_profile)
        self.assertEqual(candidate.skills_from_github, [])
        self.assertIn("No GitHub data available", candidate.data_quality_notes)


if __name__ == "__main__":
    unittest.main()
