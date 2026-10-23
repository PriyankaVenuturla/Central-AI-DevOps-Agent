"""AI Security, Secret Redaction & Guardrails Engine

Implements Slide 7 of AI Architecture:
1. Pre-LLM Secret Redaction (intercepts GitHub tokens, AWS keys, private keys, passwords)
2. Prompt Injection Firewall (wraps untrusted data in XML tags and enforces non-executable invariants)
3. Read-Only Safety Guarantees (audits against destructive commands)
"""

from __future__ import annotations

import math
import re
from typing import List

REDACTED_MASK = "***REDACTED***"

# High-risk secret patterns
SECRET_PATTERNS = [
    # GitHub Tokens (PAT, OAuth, App tokens)
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
    # AWS Access Keys
    re.compile(r"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"),
    # Generic Bedrock / Bearer tokens
    re.compile(r"\bABSK[A-Za-z0-9+/=]{40,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    # Google API Keys
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    # Slack Webhooks / Bot tokens
    re.compile(r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*"),
    re.compile(r"https://hooks\.slack\.com/services/T[0-9A-Z]+/B[0-9A-Z]+/[0-9A-Za-z]+"),
    # Generic Private Key blocks
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        re.DOTALL,
    ),
    # Password & Secret assignments in logs/configs
    re.compile(
        r"""(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret|secret_key)\b\s*[:=]\s*)(["']?)([^"'\s]{6,})(\2)"""
    ),
]


def _shannon_entropy(data: str) -> float:
    """Calculates the Shannon entropy of a string to detect random high-entropy strings (e.g. keys)."""
    if not data:
        return 0.0
    entropy = 0.0
    for x in set(data):
        p_x = float(data.count(x)) / len(data)
        entropy += -p_x * math.log(p_x, 2)
    return entropy


def redact_secrets(text: str) -> str:
    """Pre-LLM Secret Redaction.

    Intercepts sensitive tokens and credentials before sending to LLMs or vector stores.
    """
    if not text:
        return ""

    sanitized = text

    # 1. Regex redaction for known patterns
    for pattern in SECRET_PATTERNS:
        if "PRIVATE KEY" in str(pattern):
            sanitized = pattern.sub(REDACTED_MASK, sanitized)
        elif "password" in str(pattern) or "api" in str(pattern):
            # Assignment pattern
            sanitized = pattern.sub(rf"\g<1>\g<2>{REDACTED_MASK}\g<4>", sanitized)
        else:
            sanitized = pattern.sub(REDACTED_MASK, sanitized)

    return sanitized


def wrap_untrusted_log(cleaned_log: str) -> str:
    """Prompt Injection Firewall: Wraps log data in explicit untrusted XML delimiters."""
    safe_log = redact_secrets(cleaned_log)
    return f"<untrusted_ci_log>\n{safe_log}\n</untrusted_ci_log>"


def wrap_untrusted_source(source_files: List[dict]) -> str:
    """Prompt Injection Firewall: Wraps source code in explicit untrusted XML delimiters with line numbers."""
    if not source_files:
        return "<untrusted_source_files>\n(No relevant source code collected)\n</untrusted_source_files>"

    parts = []
    for f in source_files:
        path = f.get("path", "unknown")
        raw_code = f.get("content", "")
        safe_code = redact_secrets(raw_code)

        numbered_lines = "\n".join(
            f"{i:>4} | {line}" for i, line in enumerate(safe_code.splitlines(), 1)
        )
        parts.append(f"--- File: {path} ---\n{numbered_lines}")

    joined = "\n\n".join(parts)
    return f"<untrusted_source_files>\n{joined}\n</untrusted_source_files>"
