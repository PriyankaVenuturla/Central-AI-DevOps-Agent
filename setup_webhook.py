"""GitHub Webhook Automation Script

Automates creating, listing, and testing repository webhooks for CI/CD failure triage.
Subscribes to 'workflow_run' events with JSON payload and HMAC SHA-256 secret.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setup_webhook")


def list_existing_webhooks(repo: str, token: str) -> List[Dict[str, Any]]:
    """Fetches all existing webhooks on the repository."""
    url = f"https://api.github.com/repos/{repo}/hooks"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    logger.error("Failed to list webhooks for %s: HTTP %d %s", repo, resp.status_code, resp.text)
    return []


def create_github_webhook(
    repo: str,
    webhook_url: str,
    token: str,
    secret: Optional[str] = None,
    events: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Creates a repository webhook listening for workflow_run failure events."""
    if events is None:
        events = ["workflow_run"]

    # 1. Check if webhook with the same URL already exists
    existing = list_existing_webhooks(repo, token)
    for hook in existing:
        hook_url = hook.get("config", {}).get("url", "")
        if hook_url == webhook_url:
            logger.info("Webhook already exists for %s (Hook ID: %s)", webhook_url, hook["id"])
            return hook

    # 2. Create the webhook
    url = f"https://api.github.com/repos/{repo}/hooks"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    config_payload: Dict[str, Any] = {
        "url": webhook_url,
        "content_type": "json",
        "insecure_ssl": "0",
    }
    if secret:
        config_payload["secret"] = secret

    payload = {
        "name": "web",
        "active": True,
        "events": events,
        "config": config_payload,
    }

    logger.info("Registering webhook on repository '%s' -> %s...", repo, webhook_url)
    resp = requests.post(url, headers=headers, json=payload, timeout=20)

    if resp.status_code in (200, 201):
        data = resp.json()
        logger.info("Successfully created webhook! Hook ID: %s", data.get("id"))
        return data

    raise RuntimeError(
        f"Failed to create webhook (HTTP {resp.status_code}): {resp.text}"
    )


def test_ping_webhook(repo: str, hook_id: int, token: str) -> bool:
    """Sends a ping event to test connectivity."""
    url = f"https://api.github.com/repos/{repo}/hooks/{hook_id}/pings"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.post(url, headers=headers, timeout=15)
    return resp.status_code == 204


def main():
    parser = argparse.ArgumentParser(description="Create GitHub Webhook for Central-Agent CI/CD Triage")
    parser.add_argument("--repo", required=True, help="Target repository in format 'owner/repo'")
    parser.add_argument("--url", required=True, help="Public URL of the webhook receiver (e.g. https://abc.ngrok.app/webhook)")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"), help="GitHub Personal Access Token (defaults to GITHUB_TOKEN env var)")
    parser.add_argument("--secret", default=os.getenv("GITHUB_WEBHOOK_SECRET"), help="HMAC secret for payload signature verification")

    args = parser.parse_args()

    if not args.token:
        print("Error: GitHub token must be provided via --token or GITHUB_TOKEN environment variable.")
        sys.exit(1)

    try:
        hook = create_github_webhook(
            repo=args.repo,
            webhook_url=args.url,
            token=args.token,
            secret=args.secret,
        )
        print("\n" + "=" * 60)
        print("Webhook Configuration Complete")
        print("=" * 60)
        print(f"Repository : {args.repo}")
        print(f"Hook ID    : {hook.get('id')}")
        print(f"Target URL : {args.url}")
        print(f"Events     : {hook.get('events')}")
        print(f"Active     : {hook.get('active')}")
        print("=" * 60)

        # Send test ping
        if hook.get("id"):
            logger.info("Sending test ping to webhook...")
            ping_ok = test_ping_webhook(args.repo, hook["id"], args.token)
            print(f"Test Ping  : {'Success (HTTP 204)' if ping_ok else 'Failed'}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error("Failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
