"""FastAPI Webhook Listener for GitHub CI/CD Failure Events

Implements Slide 3 & 6 of AI Architecture:
1. Receives instant webhook alerts from GitHub Actions
2. Verifies HMAC SHA-256 signatures for enterprise security
3. Ignores passing builds (zero compute waste on success)
4. Dispatches failure events to the LangGraph triage pipeline asynchronously
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
import uvicorn

from config import settings
from graph.state import AgentState
from graph.workflow import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("central_agent.webhook")

app = FastAPI(
    title="Central AI DevOps Failure Triage Agent",
    description="Webhook listener that triggers AI diagnosis when GitHub Actions workflows fail.",
    version="1.0.0",
)


def verify_github_signature(payload_body: bytes, signature_header: str | None, secret: str) -> bool:
    """Verifies that the webhook payload was sent by GitHub using HMAC SHA-256."""
    if not secret:
        # Secret verification disabled in dev mode if not configured
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected_sig = signature_header[7:]
    mac = hmac.new(secret.encode("utf-8"), msg=payload_body, digestmod=hashlib.sha256)
    computed_sig = mac.hexdigest()
    return hmac.compare_digest(computed_sig, expected_sig)


def async_triage_job(state: AgentState):
    """Background task executing the LangGraph triage pipeline."""
    try:
        logger.info("Starting background triage for %s run %s", state["repo"], state["run_id"])
        final_state = run_pipeline(state)
        logger.info(
            "Completed triage for %s run %s with status: %s",
            state["repo"],
            state["run_id"],
            final_state.get("status"),
        )
    except Exception as exc:
        logger.error("Unhandled error during async triage: %s", exc, exc_info=True)


@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes probes and load balancers."""
    return {"status": "ok", "service": "Central-Agent", "model": settings.bedrock_model_id}


@app.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str = Header(None),
    x_hub_signature_256: str = Header(None),
):
    """Webhook entrypoint for GitHub Actions events."""
    body_bytes = await request.body()

    # 1. Verify cryptographic signature (only if a real non-placeholder secret is configured)
    if settings.webhook_secret and settings.webhook_secret.strip() not in ("", "your_webhook_hmac_secret_here"):
        if not verify_github_signature(body_bytes, x_hub_signature_256, settings.webhook_secret):
            logger.warning("Rejected webhook: invalid HMAC signature")
            raise HTTPException(status_code=401, detail="Invalid signature")


    # 2. Filter event type
    if x_github_event != "workflow_run":
        return {"status": "ignored", "reason": f"Unhandled event type: {x_github_event}"}

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    action = payload.get("action")
    workflow_run = payload.get("workflow_run") or {}

    conclusion = workflow_run.get("conclusion")

    # 3. Slide 1 & 3 Invariant: Run on failure only
    if action != "completed" or conclusion != "failure":
        logger.info(
            "Run %s action=%s conclusion=%s - passing or in progress, skipping (zero compute waste)",
            workflow_run.get("id"),
            action,
            conclusion,
        )
        return {
            "status": "ignored",
            "reason": f"Run conclusion is '{conclusion}', only failed runs are triaged.",
        }

    # 4. Extract triage context
    repo = payload.get("repository", {}).get("full_name") or "unknown/repo"
    run_id = str(workflow_run.get("id"))
    workflow_name = workflow_run.get("name", "Workflow")
    head_branch = workflow_run.get("head_branch", "main")
    head_sha = workflow_run.get("head_sha", "???????")

    # Extract associated pull request if any
    pull_requests = workflow_run.get("pull_requests") or []
    pr_number = pull_requests[0].get("number") if pull_requests else None

    logger.info("Accepted failure event for %s run %s (PR: %s)", repo, run_id, pr_number)

    state: AgentState = {
        "repo": repo,
        "run_id": run_id,
        "pr_number": pr_number,
        "commit_sha": head_sha,
        "branch": head_branch,
        "workflow_name": workflow_name,
        "job_name": None,
        "job_id": None,
        "raw_logs": None,
        "cleaned_log": "",
        "extracted_evidence": [],
        "source_files": [],
        "untrusted_log_xml": "",
        "untrusted_source_xml": "",
        "relevant_runbooks": [],
        "past_fixes": [],
        "analysis": None,
        "markdown_report": "",
        "status": "queued",
        "error_message": None,
    }

    # Dispatch to background task to respond promptly to GitHub
    background_tasks.add_task(async_triage_job, state)

    return Response(
        content='{"status": "accepted", "message": "Failure triage job queued"}',
        status_code=202,
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
