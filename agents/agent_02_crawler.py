import httpx
from core.config import settings
from models.candidate import GithubMetrics, CandidateSchema

#we will use async def for parallel execution 
async def crawl_github_profile(github_url: str) -> GithubMetrics | None:
    """
    Uses the GitHub API to fetch repository counts and activity.
    """
    if not github_url or "github.com" not in github_url:
        return None

    # Extract username from URL (e.g., https://github.com/user -> user)
    username = github_url.rstrip("/").split("/")[-1]
    api_url = f"https://api.github.com/users/{username}"
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {settings.GITHUB_TOKEN}" if settings.GITHUB_TOKEN else ""
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(api_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                
                # Basic enrichment logic
                return GithubMetrics(
                    repos_count=data.get("public_repos", 0),
                    total_commits=0, # Requires deeper per-repo crawling
                    code_quality_score=85.0 # Placeholder for Agent 02 scoring logic
                )
            return None
        except Exception as e:
            print(f"Agent 02 Error: {e}")
            return None

async def enrich_candidate_profile(candidate: CandidateSchema) -> CandidateSchema:
    """
    The main entry point for Agent 02 to update the candidate object.
    """
    if candidate.github_url:
        metrics = await crawl_github_profile(candidate.github_url)
        if metrics:
            candidate.github_metrics = metrics
            
    return candidate