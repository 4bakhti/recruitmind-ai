"""Tests for agents/job_fit_agent.py.

The hard-skill and composite tests are pure math — no model needed. The
semantic/end-to-end tests load the local sentence-transformers model
(first run downloads ~90 MB from Hugging Face) and are skipped if the
package is not installed.

Run from the project root:  python -m unittest tests.test_job_fit_agent
"""

import unittest

from agents import job_fit_agent
from agents.job_fit_agent import (
    HARD_SKILL_WEIGHT,
    SEMANTIC_WEIGHT,
    composite_score,
    hard_skill_match,
    score_fit,
)
from models.candidate import Candidate, WorkEntry
from models.job_description import JobDescription

try:
    import sentence_transformers  # noqa: F401
    HAVE_EMBEDDINGS = True
except Exception:  # broken/missing torch raises more than ImportError
    HAVE_EMBEDDINGS = False


FULL_MATCH_CANDIDATE = Candidate(
    name="Fits Perfectly",
    skills=["Python", "FastAPI", "PostgreSQL"],
    skills_from_github=["Docker"],
)

PARTIAL_CANDIDATE = Candidate(
    name="Missing Some",
    skills=["python"],  # lowercase on purpose — matching is normalized
    skills_from_github=[],
)

JOB = JobDescription(
    title="Backend Engineer",
    required_skills=["Python", "FastAPI", "postgres", "Docker"],
    nice_to_have_skills=["Kubernetes", "AWS"],
    description_text="Build and operate REST APIs for our hiring platform.",
)


class HardSkillMatchTests(unittest.TestCase):
    def test_all_required_skills_present(self):
        score, matched, missing = hard_skill_match(FULL_MATCH_CANDIDATE, JOB)
        self.assertEqual(score, 100.0)
        self.assertEqual(len(matched), 4)
        self.assertEqual(missing, [])

    def test_missing_skills_scored_and_listed(self):
        score, matched, missing = hard_skill_match(PARTIAL_CANDIDATE, JOB)
        self.assertEqual(score, 25.0)  # 1 of 4
        self.assertEqual(matched, ["Python"])
        # gap_analysis reports canonical names ("postgres" -> "PostgreSQL")
        self.assertEqual(missing, ["FastAPI", "PostgreSQL", "Docker"])

    def test_normalization_bridges_variants(self):
        candidate = Candidate(skills=["ReactJS", "node"])
        job = JobDescription(title="FE", required_skills=["React", "Node.js"])
        score, matched, missing = hard_skill_match(candidate, job)
        self.assertEqual(score, 100.0)
        self.assertEqual(missing, [])

    def test_github_skills_count_toward_match(self):
        candidate = Candidate(skills=[], skills_from_github=["Rust"])
        job = JobDescription(title="Systems", required_skills=["Rust"])
        score, _, _ = hard_skill_match(candidate, job)
        self.assertEqual(score, 100.0)

    def test_no_required_skills_is_vacuous_pass(self):
        score, matched, missing = hard_skill_match(
            PARTIAL_CANDIDATE, JobDescription(title="Anything Goes")
        )
        self.assertEqual(score, 100.0)
        self.assertEqual(missing, [])


class CompositeScoreTests(unittest.TestCase):
    def test_weighting_math(self):
        self.assertEqual(composite_score(100.0, 50.0), 80.0)  # .6*100+.4*50
        self.assertEqual(composite_score(0.0, 0.0), 0.0)
        self.assertEqual(composite_score(100.0, 100.0), 100.0)

    def test_weights_are_consistent(self):
        self.assertAlmostEqual(HARD_SKILL_WEIGHT + SEMANTIC_WEIGHT, 1.0)
        self.assertEqual(
            composite_score(70.0, 30.0),
            round(HARD_SKILL_WEIGHT * 70.0 + SEMANTIC_WEIGHT * 30.0, 1),
        )


@unittest.skipUnless(HAVE_EMBEDDINGS, "sentence-transformers not installed")
class EndToEndScoringTests(unittest.TestCase):
    def test_score_fit_full_result(self):
        candidate = Candidate(
            name="Jane Doe",
            skills=["Python", "FastAPI", "PostgreSQL"],
            skills_from_github=["Docker", "Rust"],
            work_history=[
                WorkEntry(
                    company="Acme GmbH",
                    title="Backend Engineer",
                    description="Built REST APIs with Python and FastAPI "
                                "for a recruiting product.",
                )
            ],
        )
        result = score_fit(candidate, JOB)

        self.assertEqual(result.hard_skill_match_score, 100.0)
        self.assertEqual(result.gap_analysis, [])
        self.assertEqual(result.nice_to_have_matches, [])
        # Same domain, so the embeddings should agree well above chance
        self.assertGreater(result.semantic_fit_score, 30.0)
        self.assertLessEqual(result.semantic_fit_score, 100.0)
        self.assertEqual(
            result.composite_score,
            composite_score(
                result.hard_skill_match_score, result.semantic_fit_score
            ),
        )
        self.assertIn("hard-skill", result.scoring_explanation)

    def test_related_job_scores_higher_than_unrelated(self):
        candidate = Candidate(
            skills=["Python", "FastAPI"],
            work_history=[
                WorkEntry(description="Built backend web services in Python.")
            ],
        )
        related = JobDescription(
            title="Python Backend Developer",
            description_text="Develop web APIs in Python.",
        )
        unrelated = JobDescription(
            title="Dental Hygienist",
            description_text="Clean teeth and assist the dentist.",
        )
        self.assertGreater(
            job_fit_agent.semantic_fit_score(candidate, related),
            job_fit_agent.semantic_fit_score(candidate, unrelated),
        )


if __name__ == "__main__":
    unittest.main()
