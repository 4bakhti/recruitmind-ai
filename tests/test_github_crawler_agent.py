"""Tests for agents/github_crawler_agent.py.

The extraction tests are offline. The crawl tests hit the real GitHub API
(unauthenticated: 60 requests/hour) and are skipped if the rate limit is
exhausted or the network is unavailable.

Run from the project root:  python -m unittest tests.test_github_crawler_agent
"""

import unittest
from datetime import date

import requests

from agents.github_crawler_agent import (
    GITHUB_API,
    enrich_candidate,
    extract_github_username,
    fetch_github_profile,
)
from models.candidate import Candidate


def _github_api_available() -> bool:
    try:
        resp = requests.get(f"{GITHUB_API}/rate_limit", timeout=5)
        return resp.ok and resp.json()["resources"]["core"]["remaining"] >= 3
    except requests.RequestException:
        return False


class ExtractUsernameTests(unittest.TestCase):
    def test_plain_url(self):
        self.assertEqual(
            extract_github_username("see github.com/torvalds for my code"),
            "torvalds",
        )

    def test_https_url_with_trailing_path(self):
        self.assertEqual(
            extract_github_username("https://github.com/octocat/Hello-World"),
            "octocat",
        )

    def test_www_and_hyphenated_username(self):
        self.assertEqual(
            extract_github_username("www.github.com/jane-doe-42/"),
            "jane-doe-42",
        )

    def test_reserved_path_is_not_a_username(self):
        self.assertIsNone(
            extract_github_username("https://github.com/features")
        )

    def test_no_github_url(self):
        self.assertIsNone(extract_github_username("just a plain CV text"))
        self.assertIsNone(extract_github_username(""))


@unittest.skipUnless(
    _github_api_available(),
    "GitHub API unreachable or unauthenticated rate limit exhausted",
)
class LiveCrawlTests(unittest.TestCase):
    def test_fetches_real_profile(self):
        profile = fetch_github_profile("torvalds")

        self.assertIsNotNone(profile)
        self.assertGreater(profile.public_repos, 0)
        self.assertGreater(profile.total_stars, 0)
        self.assertIn("C", profile.top_languages)
        self.assertTrue(profile.notable_repos)
        # linux is by far his most-starred repo
        self.assertEqual(profile.notable_repos[0].name, "linux")
        self.assertGreater(profile.notable_repos[0].stars, 1000)
        self.assertIsInstance(profile.account_created, date)
        self.assertLess(profile.account_created, date.today())
        self.assertTrue(profile.contribution_summary)

    def test_unknown_username_returns_none(self):
        profile = fetch_github_profile(
            "this-user-should-not-exist-xyz-424242"
        )
        self.assertIsNone(profile)

    def test_enrich_candidate_from_raw_text(self):
        candidate = Candidate(
            name="Linus Torvalds",
            raw_text="Kernel hacker. Code: https://github.com/torvalds",
        )
        enriched = enrich_candidate(candidate)

        self.assertEqual(enriched.github_username, "torvalds")
        self.assertIsNotNone(enriched.github_profile)
        self.assertGreater(enriched.github_profile.public_repos, 0)


class EnrichFailureTests(unittest.TestCase):
    def test_candidate_without_github_info_is_unchanged(self):
        candidate = Candidate(name="No Github", raw_text="plain CV text")
        enriched = enrich_candidate(candidate)

        self.assertIsNone(enriched.github_username)
        self.assertIsNone(enriched.github_profile)
        self.assertEqual(enriched.name, "No Github")


if __name__ == "__main__":
    unittest.main()
