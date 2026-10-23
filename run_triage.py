"""CLI Entrypoint for Direct CI/CD Failure Triage

Allows manual execution or local testing against a GitHub run ID or a local log file.
"""

from __future__ import annotations

import argparse
import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from graph.state import AgentState
from graph.workflow import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("central_agent.cli")


def main():
    parser = argparse.ArgumentParser(description="Central-Agent CI/CD Failure Triage CLI")
    parser.add_argument("--repo", default="demo-org/sample-app", help="Target GitHub repo (owner/repo)")
    parser.add_argument("--run-id", default="99999999", help="Target GitHub Actions run ID")
    parser.add_argument("--pr", type=int, default=None, help="Associated Pull Request number")
    parser.add_argument("--log-file", help="Path to local log file to triage instead of GitHub API")
    args = parser.parse_args()

    raw_logs = None
    if args.log_file:
        try:
            with open(args.log_file, "r", encoding="utf-8", errors="replace") as f:
                raw_logs = f.read()
            logger.info("Loaded %d bytes of logs from %s", len(raw_logs), args.log_file)
        except Exception as e:
            logger.error("Could not read log file: %s", e)
            sys.exit(1)

    initial_state: AgentState = {
        "repo": args.repo,
        "run_id": str(args.run_id),
        "pr_number": args.pr,
        "commit_sha": "a1b2c3d",
        "branch": "main",
        "workflow_name": "CI Build",
        "job_name": "compile-and-test",
        "job_id": 101,
        "raw_logs": raw_logs,
        "cleaned_log": "",
        "extracted_evidence": [],
        "source_files": [],
        "untrusted_log_xml": "",
        "untrusted_source_xml": "",
        "relevant_runbooks": [],
        "past_fixes": [],
        "analysis": None,
        "markdown_report": "",
        "status": "initialized",
        "error_message": None,
    }

    print("\n" + "=" * 60)
    print("🚀 Starting Central AI DevOps Triage Pipeline...")
    print(f"📦 Repository: {args.repo} | Run ID: {args.run_id}")
    print("=" * 60 + "\n")

    result = run_pipeline(initial_state)

    print("\n" + "=" * 60)
    print("📋 Generated Actionable Triage Report:")
    print("=" * 60 + "\n")
    print(result.get("markdown_report", "(No report generated)"))
    print("\n" + "=" * 60)
    print(f"🏁 Pipeline Finished. Status: {result.get('status')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
