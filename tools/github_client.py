"""GitHub API Client & Source Extractor

Provides tools to:
1. Fetch workflow run details and failed jobs
2. Download raw job logs
3. Extract source files cited in compiler/stack trace errors
4. Post structured PR comments and job summaries
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

from config import settings
from core.guardrails import redact_secrets

logger = logging.getLogger("central_agent.github")

PATH_REGEXES = [
    # General source file pattern
    re.compile(
        r"(?:^|[\s:(])([A-Za-z0-9_.\-/\\]+\.(?:java|py|js|ts|tsx|jsx|go|rs|cpp|c|h|cs|xml|yml|yaml|json|gradle|kts))(?:[:)\s]|$)",
        re.IGNORECASE,
    ),
    # Java compiler error format: File.java:[line,col]
    re.compile(r"([A-Za-z0-9_.\-/\\]+\.java):\[\d+,\d+\]", re.IGNORECASE),
    # Maven / Gradle source paths
    re.compile(r"(?:file:)?([A-Za-z0-9_.\-/\\]+/(?:src|test)/[A-Za-z0-9_.\-/\\]+)", re.IGNORECASE),
]


class GitHubClient:
    """Client for interacting with the GitHub REST API with retries and rate limit awareness."""

    def __init__(self, token: Optional[str] = None, api_url: Optional[str] = None):
        self.token = token or settings.github_token
        self.api_url = (api_url or settings.github_api_url).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Central-AI-DevOps-Agent",
        })
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.api_url}{path}"
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
            except requests.RequestException as e:
                if attempt == max_retries:
                    raise RuntimeError(f"GitHub network request failed: {e}")
                time.sleep(2**attempt)
                continue

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                time.sleep(retry_after)
                continue

            if resp.status_code >= 500 and attempt < max_retries:
                time.sleep(2**attempt)
                continue

            return resp

        raise RuntimeError("GitHub request failed after retries.")

    def get_run(self, repo: str, run_id: str) -> Dict:
        """Fetches metadata for a workflow run."""
        resp = self._request("GET", f"/repos/{repo}/actions/runs/{run_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch run {repo}/{run_id}: HTTP {resp.status_code}")
        return resp.json()

    def list_failed_jobs(self, repo: str, run_id: str) -> List[Dict]:
        """Lists all failed jobs in a workflow run."""
        resp = self._request("GET", f"/repos/{repo}/actions/runs/{run_id}/jobs?filter=latest")
        if resp.status_code != 200:
            return []
        data = resp.json()
        jobs = data.get("jobs", [])
        return [j for j in jobs if j.get("conclusion") in ("failure", "timed_out")]

    def get_job_logs(self, repo: str, job_id: int) -> Optional[str]:
        """Downloads full raw terminal logs for a job."""
        resp = self._request("GET", f"/repos/{repo}/actions/jobs/{job_id}/logs")
        if resp.status_code == 200:
            return resp.text
        return None

    def get_file_content(self, repo: str, path: str, ref: str) -> Optional[str]:
        """Fetches file contents at a specific commit SHA."""
        clean_path = path.lstrip("/")
        resp = self._request("GET", f"/repos/{repo}/contents/{clean_path}?ref={ref}")
        if resp.status_code != 200:
            return None

        data = resp.json()
        if data.get("type") != "file":
            return None

        raw_content = data.get("content", "")
        if not raw_content:
            return None

        try:
            return base64.b64decode(raw_content).decode("utf-8", errors="replace")
        except Exception:
            return None

    def post_pr_comment(self, repo: str, pr_number: int, markdown_body: str) -> bool:
        """Approach 1: Posts a markdown triage comment to a Pull Request."""
        safe_body = redact_secrets(markdown_body)
        resp = self._request(
            "POST",
            f"/repos/{repo}/issues/{pr_number}/comments",
            json={"body": safe_body},
        )
        return resp.status_code in (200, 201)

    def create_check_run_summary(
        self,
        repo: str,
        head_sha: str,
        title: str,
        summary_markdown: str,
    ) -> bool:
        """Approach 2: Posts a rich Check Run Summary directly onto the GitHub Actions workflow run page."""
        safe_summary = redact_secrets(summary_markdown)
        payload = {
            "name": "AI Failure Triage Summary",
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": "neutral",
            "output": {
                "title": title[:250],
                "summary": safe_summary[:65000],
            },
        }
        try:
            resp = self._request("POST", f"/repos/{repo}/check-runs", json=payload)
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.warning("Could not create check-run summary for %s commit %s: %s", repo, head_sha, e)
            return False

    def post_commit_comment(self, repo: str, commit_sha: str, markdown_body: str) -> bool:
        """Posts a direct commit-level analysis comment when no PR exists."""
        safe_body = redact_secrets(markdown_body)
        try:
            resp = self._request(
                "POST",
                f"/repos/{repo}/commits/{commit_sha}/comments",
                json={"body": safe_body},
            )
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.warning("Could not post commit comment for %s %s: %s", repo, commit_sha, e)
            return False


    @staticmethod
    def extract_source_paths_from_logs(logs: str) -> List[str]:
        """Scans logs for file path mentions that likely caused the error."""
        candidates = set()
        for pattern in PATH_REGEXES:
            for match in pattern.finditer(logs):
                raw_path = match.group(1).replace("\\", "/")
                # Strip drive letters (e.g. C:/)
                raw_path = re.sub(r"^[A-Za-z]:/", "", raw_path)
                if "://" in raw_path or "node_modules" in raw_path or ".git" in raw_path:
                    continue
                candidates.add(raw_path.lstrip("/"))
        return list(candidates)[: settings.max_source_files]
