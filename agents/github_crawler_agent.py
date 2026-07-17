"""GitHub Crawler Agent.

Standalone enrichment agent: given a GitHub username (or a Candidate whose
raw_text contains a GitHub URL), fetches public profile + repo data from the
GitHub REST API and populates Candidate.github_profile.

Works unauthenticated (60 requests/hour); set GITHUB_TOKEN in .env to raise
the limit to 5000/hour.
"""

import logging
import os
import re
from datetime import date, datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from models.candidate import Candidate, GithubProfile, NotableRepo

load_dotenv()

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT = 10  # seconds

# github.com/<x> paths that are never user profiles
_RESERVED_PATHS = {
    "orgs", "features", "topics", "collections", "sponsors", "marketplace",
    "apps", "settings", "login", "join", "enterprise", "about", "pricing",
    "site", "explore", "search", "trending", "notifications", "issues",
    "pulls", "new", "organizations", "contact", "security",
}

# GitHub username rules: alphanumeric + single hyphens, no leading/trailing
# hyphen, max 39 chars
_GITHUB_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/"
    r"([A-Za-z\d](?:[A-Za-z\d]|-(?=[A-Za-z\d])){0,38})",
    re.IGNORECASE,
)


def extract_github_username(text: str) -> str | None:
    """Pull a GitHub username out of free text containing a github.com URL."""
    if not text:
        return None
    for match in _GITHUB_URL_RE.finditer(text):
        username = match.group(1)
        if username.lower() not in _RESERVED_PATHS:
            return username
    return None


def _headers(with_token: bool = True) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token and with_token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(url: str, params: dict | None = None) -> requests.Response | None:
    """GET with clear logging for the failure modes we care about."""
    try:
        response = requests.get(
            url, headers=_headers(), params=params, timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 401 and os.getenv("GITHUB_TOKEN"):
            # Invalid/expired GITHUB_TOKEN — retry unauthenticated rather
            # than failing outright (60 req/hour limit applies)
            logger.warning(
                "GITHUB_TOKEN was rejected by GitHub (401) — retrying "
                "without auth. Fix or remove the token in .env."
            )
            response = requests.get(
                url, headers=_headers(with_token=False), params=params,
                timeout=REQUEST_TIMEOUT,
            )
    except requests.RequestException as e:
        logger.error("GitHub API request failed for %s: %s", url, e)
        return None

    if response.status_code == 404:
        logger.warning("GitHub API 404 (not found): %s", url)
        return None
    if response.status_code in (403, 429):
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            logger.warning(
                "GitHub API rate limit exceeded (set GITHUB_TOKEN in .env "
                "to raise it): %s", url,
            )
        else:
            logger.warning("GitHub API access forbidden (%s): %s",
                           response.status_code, url)
        return None
    if not response.ok:
        logger.warning("GitHub API error %s: %s", response.status_code, url)
        return None
    return response


def fetch_github_profile(username: str) -> GithubProfile | None:
    """Fetch and summarize a user's public GitHub presence.

    Returns None on any failure (unknown user, rate limit, network error).
    """
    user_resp = _get(f"{GITHUB_API}/users/{username}")
    if user_resp is None:
        return None
    user = user_resp.json()

    repos_resp = _get(
        f"{GITHUB_API}/users/{username}/repos",
        params={"per_page": 100, "sort": "pushed", "direction": "desc"},
    )
    repos = repos_resp.json() if repos_resp is not None else []

    # Top languages by how many repos use them
    language_counts: dict[str, int] = {}
    for repo in repos:
        if repo.get("language"):
            language_counts[repo["language"]] = (
                language_counts.get(repo["language"], 0) + 1
            )
    top_languages = sorted(
        language_counts, key=language_counts.get, reverse=True
    )[:5]

    by_stars = sorted(
        repos, key=lambda r: r.get("stargazers_count", 0), reverse=True
    )
    notable_repos = [
        NotableRepo(
            name=repo.get("name"),
            description=repo.get("description"),
            stars=repo.get("stargazers_count", 0),
            language=repo.get("language"),
            last_updated=repo.get("pushed_at"),
        )
        for repo in by_stars[:5]
    ]

    total_stars = sum(r.get("stargazers_count", 0) for r in repos)

    account_created = None
    if user.get("created_at"):
        # "2011-09-03T15:26:22Z" -> date
        account_created = date.fromisoformat(user["created_at"][:10])

    # The REST API doesn't expose the contribution graph (that needs GraphQL
    # + auth), so summarize recent activity from repo pushes instead.
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    pushed_recently = sum(
        1 for r in repos
        if r.get("pushed_at")
        and datetime.fromisoformat(r["pushed_at"].replace("Z", "+00:00"))
        > one_year_ago
    )
    activity = "Active" if pushed_recently else "Quiet"
    contribution_summary = (
        f"{activity} — {pushed_recently} of {len(repos)} public repos "
        f"pushed to in the last year"
    )

    return GithubProfile(
        public_repos=user.get("public_repos", 0),
        top_languages=top_languages,
        notable_repos=notable_repos,
        total_stars=total_stars,
        account_created=account_created,
        contribution_summary=contribution_summary,
    )


def enrich_candidate(candidate: Candidate) -> Candidate:
    """Populate candidate.github_username / github_profile if possible.

    Never raises; on any failure the candidate is returned unchanged with
    github_profile left as None.
    """
    username = candidate.github_username
    if not username and candidate.raw_text:
        username = extract_github_username(candidate.raw_text)

    if not username:
        logger.info("No GitHub username found for candidate %s — skipping",
                    candidate.name or "<unknown>")
        return candidate

    profile = fetch_github_profile(username)
    if profile is None:
        logger.info("Could not fetch GitHub profile for %r — leaving "
                    "candidate unchanged", username)
        return candidate

    candidate.github_username = username
    candidate.github_profile = profile
    return candidate
