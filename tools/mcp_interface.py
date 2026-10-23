"""Model Context Protocol (MCP) Tool Abstractions

Implements Slide 5 of AI Architecture:
Standard Tool Connections across Repositories, Vector Database, and Notifications.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from memory.chroma_memory import ChromaMemoryManager
from tools.github_client import GitHubClient


class MCPToolRegistry:
    """Central registry conforming to MCP tool call conventions."""

    def __init__(
        self,
        github_client: Optional[GitHubClient] = None,
        memory_manager: Optional[ChromaMemoryManager] = None,
    ):
        self.github = github_client or GitHubClient()
        self.memory = memory_manager or ChromaMemoryManager()

    # --- GitHub Tools ---
    def get_job_logs(self, repo: str, job_id: int) -> Optional[str]:
        """MCP Tool: Fetches build logs from a GitHub repository."""
        return self.github.get_job_logs(repo, job_id)

    def get_file_content(self, repo: str, path: str, commit_sha: str) -> Optional[str]:
        """MCP Tool: Fetches specific source code from the exact commit."""
        return self.github.get_file_content(repo, path, commit_sha)

    def post_pr_comment(self, repo: str, pr_number: int, markdown_report: str) -> bool:
        """MCP Tool: Publishes the triage report to the GitHub PR."""
        return self.github.post_pr_comment(repo, pr_number, markdown_report)

    # --- Vector DB Tools ---
    def search_past_fixes(self, error_signature: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """MCP Tool: Searches team historical fixes in ChromaDB using all-MiniLM-L6-v2."""
        return self.memory.search_past_fixes(error_signature, n_results=top_k)

    def get_team_runbook(self, category_or_error: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """MCP Tool: Retrieves team operational SOPs from ChromaDB."""
        return self.memory.search_runbooks(category_or_error, n_results=top_k)

    def save_new_fix(self, repo: str, signature: str, root_cause: str, fix: str):
        """MCP Tool: Records newly verified solutions into ChromaDB institutional memory."""
        self.memory.save_resolved_fix(signature, root_cause, fix, repo=repo)
