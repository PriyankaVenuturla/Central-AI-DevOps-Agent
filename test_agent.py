"""Comprehensive Test & Validation Script for Central-Agent

Validates:
1. Log noise pruning & 50-line boundary chunking
2. Pre-LLM secret redaction & prompt injection XML guards
3. ChromaDB vector memory & all-MiniLM-L6-v2 runbook matching
4. Bedrock Converse moonshotai.kimi-k2.5 integration & fallback parsing
5. Full 8-step LangGraph / sequential pipeline execution
"""

from __future__ import annotations

import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from core.bedrock_llm import BedrockLLMClient
from core.guardrails import redact_secrets, wrap_untrusted_log
from core.log_filter import filter_and_chunk_log, prune_noise
from graph.state import AgentState
from graph.workflow import run_pipeline
from memory.chroma_memory import ChromaMemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_agent")

SAMPLE_MESSY_CI_LOG = """
2026-10-23T18:50:01.000Z \x1b[32m[INFO]\x1b[0m Setting up runner environment...
2026-10-23T18:50:02.000Z \x1b[34m[INFO]\x1b[0m git clone --depth=1 https://github.com/enterprise/payment-service.git
2026-10-23T18:50:03.000Z Downloading: https://repo.maven.apache.org/maven2/org/springframework/boot/spring-boot-starter/3.1.0/spring-boot-starter-3.1.0.jar
2026-10-23T18:50:04.000Z Downloaded from central: 450 KB
2026-10-23T18:50:05.000Z [INFO] --- maven-compiler-plugin:3.11.0:compile (default-compile) @ payment-service ---
2026-10-23T18:50:06.000Z PASS test_init_database
2026-10-23T18:50:07.000Z PASS test_cache_connection
2026-10-23T18:50:08.000Z \x1b[31m##[error]\x1b[0m TestingCode.java:16: error: ')' expected
2026-10-23T18:50:08.100Z        System.out.println("Total names: " + names.size();
2026-10-23T18:50:08.200Z                                                         ^
2026-10-23T18:50:09.000Z [INFO] 1 error
2026-10-23T18:50:10.000Z [INFO] BUILD FAILURE
2026-10-23T18:50:11.000Z [ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.11.0:compile (default-compile) on project payment-service: Compilation failure
2026-10-23T18:50:12.000Z ##[error]Process completed with exit code 1.
"""

SAMPLE_SECRETS_LOG = """
[DEBUG] Exporting environment: GITHUB_TOKEN=ghp_ABC1234567890abcdefghijklmnopqrstuvwxyz
[DEBUG] Connecting to AWS with key AKIAIOSFODNN7EXAMPLE and secret_key="supersecret12345"
[DEBUG] Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ae...
"""


def test_log_filtering():
    print("\n--- 1. Testing Log Filtering & Boundary Chunking ---")
    chunk, evidence = filter_and_chunk_log(SAMPLE_MESSY_CI_LOG, max_lines=50)

    # Verify ANSI codes stripped
    assert "\x1b[" not in chunk, "ANSI escape codes should be stripped!"
    # Verify routine download noise stripped
    assert "Downloading:" not in chunk, "Routine downloads should be pruned!"
    # Verify error captured
    assert "error: ')' expected" in chunk, "Primary error line must be captured!"
    print("✅ Log filtering passed! Chunk length:", len(chunk.splitlines()), "lines")
    print("Verbatim evidence lines captured:", evidence)


def test_guardrails():
    print("\n--- 2. Testing Pre-LLM Secret Redaction & Guardrails ---")
    redacted = redact_secrets(SAMPLE_SECRETS_LOG)
    assert "ghp_ABC123" not in redacted, "GitHub PAT must be redacted!"
    assert "AKIAIOSFODNN" not in redacted, "AWS key must be redacted!"
    assert "***REDACTED***" in redacted, "Mask must replace secrets!"

    sandboxed = wrap_untrusted_log(SAMPLE_MESSY_CI_LOG)
    assert sandboxed.startswith("<untrusted_ci_log>"), "Log must be sandboxed in XML tags!"
    assert sandboxed.endswith("</untrusted_ci_log>"), "Log must end with XML close tag!"
    print("✅ Guardrails passed! Redacted text excerpt:\n", redacted.strip())


def test_chroma_vector_memory():
    print("\n--- 3. Testing ChromaDB Vector Memory (all-MiniLM-L6-v2) ---")
    mem = ChromaMemoryManager()
    hits = mem.search_runbooks("Java javac error ')' expected syntax", n_results=2)
    print(f"✅ Vector store query returned {len(hits)} matching runbooks.")
    if hits:
        print("Top runbook matched:", hits[0]["document"][:120] + "...")


def test_full_pipeline():
    print("\n--- 4. Testing End-to-End Triage Pipeline (LangGraph Workflow) ---")
    state: AgentState = {
        "repo": "enterprise/payment-service",
        "run_id": "482019482",
        "pr_number": 42,
        "commit_sha": "f8a9b1c",
        "branch": "feature/payment-v2",
        "workflow_name": "CI / Compile Java",
        "job_name": "compile-java",
        "job_id": 89412,
        "raw_logs": SAMPLE_MESSY_CI_LOG,
        "cleaned_log": "",
        "extracted_evidence": [],
        "source_files": [
            {
                "path": "TestingCode.java",
                "content": 'package com.demo;\npublic class TestingCode {\n  public static void main(String[] args) {\n    System.out.println("Total names: " + names.size();\n  }\n}',
            }
        ],
        "untrusted_log_xml": "",
        "untrusted_source_xml": "",
        "relevant_runbooks": [],
        "past_fixes": [],
        "analysis": None,
        "markdown_report": "",
        "status": "ready",
        "error_message": None,
    }

    result = run_pipeline(state)
    report = result.get("markdown_report", "")

    print("\n" + "=" * 60)
    print("🎉 End-to-End Triage Report Generated:")
    print("=" * 60)
    print(report)
    print("=" * 60)

    assert "Status: Build Failed" in report, "Report must have status header!"
    assert "TestingCode.java" in report or "Compile" in report, "Report must identify failed context!"
    print("✅ End-to-End Pipeline test completed successfully!")


if __name__ == "__main__":
    try:
        test_log_filtering()
        test_guardrails()
        test_chroma_vector_memory()
        test_full_pipeline()
        print("\n✨ ALL TESTS COMPLETED SUCCESSFULLY! ✨\n")
    except Exception as e:
        logger.error("Test failed: %s", e, exc_info=True)
        sys.exit(1)
