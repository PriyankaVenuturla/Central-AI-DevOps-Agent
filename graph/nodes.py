"""LangGraph Node Implementations for CI/CD Failure Triage

Decomposes the triage lifecycle into discrete, testable nodes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from core.bedrock_llm import BedrockLLMClient
from core.guardrails import redact_secrets, wrap_untrusted_log, wrap_untrusted_source
from core.log_filter import filter_and_chunk_log
from graph.state import AgentState
from memory.chroma_memory import ChromaMemoryManager
from models import AnalysisResult
from tools.github_client import GitHubClient

logger = logging.getLogger("central_agent.nodes")

# Singletons for node reuse
github_client = GitHubClient()
chroma_memory = ChromaMemoryManager()
bedrock_llm = BedrockLLMClient()


def fetch_logs_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Fetches job logs and metadata from GitHub if not provided directly."""
    logger.info("Executing fetch_logs_node for %s run %s", state["repo"], state["run_id"])

    # If raw_logs was already passed (e.g. from webhook or CLI simulation), keep it
    if state.get("raw_logs"):
        return {"status": "logs_ready"}

    try:
        failed_jobs = github_client.list_failed_jobs(state["repo"], state["run_id"])
        if not failed_jobs:
            logger.warning("No failed jobs found for run %s", state["run_id"])
            return {
                "raw_logs": "(No failed jobs returned by GitHub Actions API)",
                "job_name": "unknown",
                "status": "no_failed_jobs",
            }

        # Focus on the first failed job
        target_job = failed_jobs[0]
        job_id = target_job.get("id")
        job_name = target_job.get("name", "build")

        raw_logs = github_client.get_job_logs(state["repo"], job_id) or ""
        return {
            "raw_logs": raw_logs,
            "job_name": job_name,
            "job_id": job_id,
            "status": "logs_fetched",
        }
    except Exception as exc:
        logger.error("Error fetching logs from GitHub: %s", exc)
        return {
            "raw_logs": f"Error fetching logs: {exc}",
            "job_name": "build",
            "status": "fetch_error",
            "error_message": str(exc),
        }


def filter_logs_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Executes noise pruning (regex) and 50-line boundary chunking."""
    logger.info("Executing filter_logs_node...")
    raw = state.get("raw_logs") or ""

    cleaned_chunk, evidence = filter_and_chunk_log(raw, max_lines=50)

    return {
        "cleaned_log": cleaned_chunk,
        "extracted_evidence": evidence,
        "status": "logs_filtered",
    }


def redact_and_sandbox_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Redacts secrets before AI processing and applies XML prompt injection firewalls."""
    logger.info("Executing redact_and_sandbox_node...")
    cleaned = state.get("cleaned_log") or ""

    untrusted_log_xml = wrap_untrusted_log(cleaned)

    return {
        "untrusted_log_xml": untrusted_log_xml,
        "status": "redacted_and_sandboxed",
    }


def collect_source_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Extracts source file paths from the log excerpt and retrieves commit files."""
    logger.info("Executing collect_source_node...")
    cleaned = state.get("cleaned_log") or ""
    repo = state.get("repo")
    commit_sha = state.get("commit_sha")

    source_files: List[Dict[str, str]] = []

    if repo and commit_sha and commit_sha != "???????":
        paths = GitHubClient.extract_source_paths_from_logs(cleaned)
        logger.info("Identified candidate source paths: %s", paths)

        for path in paths:
            try:
                content = github_client.get_file_content(repo, path, commit_sha)
                if content:
                    source_files.append({"path": path, "content": content})
            except Exception as e:
                logger.warning("Could not fetch source file %s: %s", path, e)

    untrusted_source_xml = wrap_untrusted_source(source_files)

    return {
        "source_files": source_files,
        "untrusted_source_xml": untrusted_source_xml,
        "status": "source_collected",
    }


def vector_lookup_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Queries ChromaDB for similar past fixes and team runbooks using all-MiniLM-L6-v2."""
    logger.info("Executing vector_lookup_node...")
    evidence = state.get("extracted_evidence") or []
    query_text = " ".join(evidence) if evidence else state.get("cleaned_log", "")[:500]

    past_fixes = chroma_memory.search_past_fixes(query_text, n_results=2)
    runbooks = chroma_memory.search_runbooks(query_text, n_results=2)

    logger.info("Vector store returned %d past fixes, %d runbooks", len(past_fixes), len(runbooks))

    return {
        "past_fixes": past_fixes,
        "relevant_runbooks": runbooks,
        "status": "memory_retrieved",
    }


def llm_reasoning_node(state: AgentState) -> Dict[str, Any]:
    """Node 6: Calls AWS Bedrock with moonshotai.kimi-k2.5 to diagnose the failure."""
    logger.info("Executing llm_reasoning_node with Bedrock moonshotai.kimi-k2.5...")
    job_name = state.get("job_name") or "CI Build"
    failed_step = "Compile / Test"

    # Format historical context for prompt
    past_contexts = []
    for fix in state.get("past_fixes", []):
        past_contexts.append(f"Past Incident: {fix.get('document')}")
    for rb in state.get("relevant_runbooks", []):
        past_contexts.append(f"Runbook SOP: {rb.get('document')}")

    history_str = "\n\n".join(past_contexts)

    analysis = bedrock_llm.analyze_failure(
        job_name=job_name,
        failed_step=failed_step,
        cleaned_log_xml=state.get("untrusted_log_xml", ""),
        source_context_xml=state.get("untrusted_source_xml", ""),
        past_fixes_context=history_str,
    )

    return {
        "analysis": analysis,
        "status": "reasoning_complete",
    }


def format_report_node(state: AgentState) -> Dict[str, Any]:
    """Node 7: Builds the clean, actionable 15-second markdown report (Slide 8)."""
    logger.info("Executing format_report_node...")
    a = state.get("analysis")
    if not a:
        a = AnalysisResult(
            failed_job=state.get("job_name", "build"),
            failed_step="unknown",
            category="unknown",
            confidence="low",
            root_cause="Analysis could not be generated.",
            suggested_fix="Check CI logs manually.",
            evidence=state.get("extracted_evidence", []),
        )

    # Format evidence code block
    evidence_text = "\n".join(a.evidence) if a.evidence else "(No direct evidence lines available)"

    lines = [
        f"<!-- central-agent run:{state.get('run_id')} -->",
        f"### 🔴 Status: Build Failed | Root Cause Identified ({a.confidence.capitalize()} Confidence)",
        "",
        f"**Failed Step:** `{a.failed_job} / {a.failed_step}`  ",
        f"**Error Category:** `{a.category.replace('_', ' ').capitalize()}`  ",
        "",
        "#### 📋 Plain English Root Cause",
        f"{a.root_cause}",
        "",
        "#### 🔍 Exact Error in Log",
        "```text",
        evidence_text,
        "```",
        "",
        "#### 💡 Suggested Fix",
        f"{a.suggested_fix}",
    ]

    # Format C: Side-by-side Code Comparison Table & Unified Diff
    if a.broken_code or a.fixed_code or a.diff_patch:
        lines += [
            "",
            "#### 🛠️ Expected Code Changes (Side-by-Side Comparison)",
        ]
        if a.file_path:
            line_str = f" (Line {a.line_number})" if a.line_number else ""
            lines += [f"**Target File:** `{a.file_path}`{line_str}", ""]

        # Side-by-side Table
        if a.broken_code and a.fixed_code:
            lines += [
                "| ❌ Broken Line (Current Commit) | ✅ Corrected Line (Suggested Fix) |",
                "|---|---|",
                f"| `{a.broken_code.strip()}` | `{a.fixed_code.strip()}` |",
                "",
            ]

        # Unified Diff
        if a.diff_patch:
            clean_diff = a.diff_patch.strip()
            if not clean_diff.startswith("```"):
                lines += ["```diff", clean_diff, "```", ""]
            else:
                lines += [clean_diff, ""]
        elif a.broken_code and a.fixed_code:
            target_path = a.file_path or "source_code"
            ln = a.line_number or 1
            lines += [
                "```diff",
                f"--- a/{target_path}",
                f"+++ b/{target_path}",
                f"@@ -{ln},1 +{ln},1 @@",
                f"- {a.broken_code.strip()}",
                f"+ {a.fixed_code.strip()}",
                "```",
                "",
            ]

    if a.source_analysis:
        lines += [
            "",
            "#### 📂 Source Code Verification",
            f"{a.source_analysis}",
        ]

    if a.likely_followup_errors:
        lines += [
            "",
            "#### ⚠️ Likely Follow-up Errors (Unreported by Compiler)",
            *[f"- `{e}`" for e in a.likely_followup_errors],
        ]

    lines += [
        "",
        "---",
        "<sub>*AI DevOps Agent: Triage and Alert only. Production code was not modified.*</sub>",
    ]

    report_markdown = redact_secrets("\n".join(lines))
    return {
        "markdown_report": report_markdown,
        "status": "report_formatted",
    }


def publish_comment_node(state: AgentState) -> Dict[str, Any]:
    """Node 8: Posts triage report to Approach 1 (PR Comment) & Approach 2 (Actions Run Summary)."""
    logger.info("Executing publish_comment_node (Approach 1 & 2)...")
    report = state.get("markdown_report", "")
    repo = state.get("repo")
    pr_number = state.get("pr_number")
    commit_sha = state.get("commit_sha")
    analysis = state.get("analysis")
    category = analysis.category if analysis else "Build Failure"

    # =========================================================================
    # APPROACH 1: Pull Request Comment (Where developers fix code)
    # =========================================================================
    if repo and pr_number:
        try:
            success = github_client.post_pr_comment(repo, pr_number, report)
            logger.info("Approach 1 (PR Comment): Posted to %s#%s: %s", repo, pr_number, success)
        except Exception as e:
            logger.warning("Approach 1 (PR Comment) failed on %s#%s: %s", repo, pr_number, e)

    # =========================================================================
    # APPROACH 2: GitHub Actions Run Summary (Where developers view run)
    # =========================================================================
    # 2a. Post rich Check Run Summary on GitHub Actions run page via Checks API
    if repo and commit_sha and commit_sha != "???????":
        try:
            title = f"AI Triage: {category.replace('_', ' ').capitalize()} Identified"
            check_ok = github_client.create_check_run_summary(
                repo=repo,
                head_sha=commit_sha,
                title=title,
                summary_markdown=report,
            )
            logger.info("Approach 2 (Check Run Summary): Created on %s commit %s: %s", repo, commit_sha, check_ok)
        except Exception as e:
            logger.warning("Approach 2 (Check Run Summary) failed: %s", e)

        # 2b. If no PR exists, also post direct commit-level comment
        if not pr_number:
            try:
                commit_ok = github_client.post_commit_comment(repo, commit_sha, report)
                logger.info("Approach 2 (Commit Comment): Posted on %s commit %s: %s", repo, commit_sha, commit_ok)
            except Exception as e:
                logger.warning("Approach 2 (Commit Comment) failed: %s", e)

    # 2c. Write to GITHUB_STEP_SUMMARY if running inside a runner
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path and os.path.exists(os.path.dirname(summary_path)):
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n{report}\n")
            logger.info("Approach 2 (GITHUB_STEP_SUMMARY): Written to %s", summary_path)
        except Exception as e:
            logger.warning("Approach 2 (GITHUB_STEP_SUMMARY) failed: %s", e)

    return {"status": "completed"}

