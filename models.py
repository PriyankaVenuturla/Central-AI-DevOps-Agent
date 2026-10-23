"""Data Models and Pydantic Schemas for Central-Agent."""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Structured root cause analysis produced by the LLM."""

    failed_job: str = Field(description="Name of the job that failed")
    failed_step: str = Field(description="Specific step or command that failed")
    category: str = Field(
        default="compilation",
        description="Category: compilation, syntax_error, dependency, test_failure, configuration, infrastructure, permissions, timeout, flaky, or unknown",
    )
    confidence: str = Field(
        default="high",
        description="Confidence level: high, medium, or low",
    )
    root_cause: str = Field(
        description="1-3 clear sentences explaining the primary root cause in plain English"
    )
    suggested_fix: str = Field(
        description="Actionable, concrete steps for the developer to resolve the issue"
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="3-8 verbatim error lines extracted from the CI log",
    )
    source_analysis: Optional[str] = Field(
        default=None,
        description="Explanation of how the committed source code confirms or explains the error",
    )
    likely_followup_errors: List[str] = Field(
        default_factory=list,
        description="Errors hidden behind syntax/parse failures (e.g. unimported types or missing symbols)",
    )
    # Format C: Side-by-side code diff fields
    file_path: Optional[str] = Field(
        default=None,
        description="Exact file path where the error occurred (e.g., src/main/java/com/demo/TestingCode.java)",
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Line number where the error occurred",
    )
    broken_code: Optional[str] = Field(
        default=None,
        description="The exact broken or problematic line(s) of code from the current commit",
    )
    fixed_code: Optional[str] = Field(
        default=None,
        description="The exact corrected line(s) of code that fixes the problem",
    )
    diff_patch: Optional[str] = Field(
        default=None,
        description="Unified diff patch representation with - and + markers",
    )



class JobLogData(BaseModel):
    """Normalized metadata and log excerpt for a failed job."""

    name: str
    job_id: int
    url: str
    failed_steps: List[str] = Field(default_factory=list)
    raw_excerpt: str = ""
    cleaned_excerpt: str = ""
    logs_available: bool = True


class SourceFile(BaseModel):
    """Source file content extracted from the failed commit."""

    path: str
    content: str


class WebhookRunEvent(BaseModel):
    """Parsed GitHub workflow_run event."""

    repo: str
    run_id: str
    workflow_name: str
    branch: str
    commit_sha: str
    run_url: str
    sender: str
    pr_number: Optional[int] = None
    conclusion: Optional[str] = None
