"""Log Filtering & Boundary Chunking Engine

Implements Slide 4 of AI Architecture:
1. Noise Pruning (Regex): Strips ANSI codes, ISO timestamps, package download tickers, and PASS lines.
2. Boundary Chunking: Scans for failure markers ('error:', 'Exception in thread', 'Traceback')
   and extracts a compact ~50-line perimeter around the failure epicenter.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ANSI color codes
ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# ISO Timestamps (e.g. 2026-10-23T18:59:40.123456Z or 2026-10-23 18:59:40 )
TIMESTAMP_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\s*",
    re.MULTILINE,
)

# Routine noise patterns that consume tokens without providing failure insight
ROUTINE_NOISE_PATTERNS = [
    re.compile(r"^\s*(?:npm|yarn|pnpm)\s+(?:http|verb|sill|info)\s+fetch", re.IGNORECASE),
    re.compile(r"^\s*(?:Downloading|Downloaded|Extracting|Unpacking|Cached):?\s+.*", re.IGNORECASE),
    re.compile(r"^\s*\[INFO\]\s+---.*---$", re.IGNORECASE),
    re.compile(r"^\s*\[INFO\]\s+Downloaded from.*", re.IGNORECASE),
    re.compile(r"^\s*(?:PASS|ok|PASSED)\s+.*", re.IGNORECASE),
    re.compile(r"^\s*✓\s+.*"),
    re.compile(r"^\s*\d+%\s*\[.*\]\s*"),
]

# Failure markers (anchors) indicating where the root cause or stack trace begins
FAILURE_MARKERS = [
    re.compile(r"##\[error\]", re.IGNORECASE),
    re.compile(r"\b(?:fatal|panic|failure|failed)\b", re.IGNORECASE),
    re.compile(r"\bException in thread\b", re.IGNORECASE),
    re.compile(r"\bTraceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r"\bBUILD FAILED\b", re.IGNORECASE),
    re.compile(r"\berror:\s+", re.IGNORECASE),
    re.compile(r"\bSyntaxError\b", re.IGNORECASE),
    re.compile(r"\bNullPointerException\b", re.IGNORECASE),
    re.compile(r"\bFAIL:\s+", re.IGNORECASE),
    re.compile(r"\bexit code [1-9]\b", re.IGNORECASE),
    re.compile(r"\bnpm ERR!\b", re.IGNORECASE),
    re.compile(r"FAILED\s+\(failures=\d+\)", re.IGNORECASE),
]


def prune_noise(raw_logs: str) -> str:
    """Removes ANSI formatting, timestamps, and routine download/pass lines.

    Cuts out 80% to 90% of clutter.
    """
    if not raw_logs:
        return ""

    # 1. Strip ANSI escape sequences
    cleaned = ANSI_REGEX.sub("", raw_logs)

    # 2. Strip timestamps
    cleaned = TIMESTAMP_REGEX.sub("", cleaned)

    # 3. Filter lines matching routine progress or passing items
    filtered_lines: List[str] = []
    for line in cleaned.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if any(p.search(line_stripped) for p in ROUTINE_NOISE_PATTERNS):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def find_failure_indices(lines: List[str]) -> List[int]:
    """Scans lines for error markers and returns line indices of hits."""
    indices: List[int] = []
    for idx, line in enumerate(lines):
        if any(marker.search(line) for marker in FAILURE_MARKERS):
            indices.append(idx)
    return indices


def extract_boundary_chunk(
    lines: List[str],
    anchor_indices: List[int],
    max_lines: int = 50,
) -> Tuple[str, List[str]]:
    """Extracts a bounded chunk (~50 lines) around the principal failure anchor.

    Returns (chunk_str, error_evidence_lines).
    """
    if not lines:
        return "(empty log)", []

    if not anchor_indices:
        # Fallback to the tail of the log if no specific anchor was matched
        selected = lines[-max_lines:]
        return "\n".join(selected), selected[-5:]

    # Pick the most definitive failure anchor (usually first severe error or compiler fault)
    primary_anchor = anchor_indices[0]

    # Calculate ~50 line window centered around the error anchor
    half = max_lines // 2
    start_idx = max(0, primary_anchor - (half // 2))
    end_idx = min(len(lines), start_idx + max_lines)

    chunk_lines = lines[start_idx:end_idx]

    # Gather verbatim evidence lines
    evidence = [lines[idx].strip() for idx in anchor_indices if start_idx <= idx < end_idx]
    if not evidence and chunk_lines:
        evidence = [chunk_lines[0].strip()]

    return "\n".join(chunk_lines), evidence[:8]


def filter_and_chunk_log(
    raw_logs: str,
    max_lines: int = 50,
) -> Tuple[str, List[str]]:
    """Complete 2-stage log filter:

    Stage 1: Noise Pruning (Regex)
    Stage 2: Boundary Chunking (50 lines around failure markers)

    Returns (cleaned_50_line_excerpt, list_of_error_lines).
    """
    pruned = prune_noise(raw_logs)
    lines = pruned.splitlines()

    anchor_indices = find_failure_indices(lines)
    return extract_boundary_chunk(lines, anchor_indices, max_lines=max_lines)
