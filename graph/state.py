"""Agent State definition for LangGraph triage workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from models import AnalysisResult


class AgentState(TypedDict):
    """The central state dictionary passed between LangGraph nodes."""

    # Workflow Identity
    repo: str
    run_id: str
    pr_number: Optional[int]
    commit_sha: Optional[str]
    branch: Optional[str]
    workflow_name: Optional[str]
    job_name: Optional[str]
    job_id: Optional[int]

    # Raw Inputs & Logs
    raw_logs: Optional[str]
    cleaned_log: str
    extracted_evidence: List[str]

    # Context & Sandboxing
    source_files: List[Dict[str, str]]
    untrusted_log_xml: str
    untrusted_source_xml: str

    # Vector Memory (ChromaDB)
    relevant_runbooks: List[Dict[str, Any]]
    past_fixes: List[Dict[str, Any]]

    # AI Reasoning & Output
    analysis: Optional[AnalysisResult]
    markdown_report: str

    # Execution Meta
    status: str
    error_message: Optional[str]
