"""LangGraph Workflow Assembler & Pipeline Runner

Assembles the StateGraph connecting all 8 triage steps into an executable graph.
Includes seamless fallback execution if langgraph is not yet installed in the environment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from graph.nodes import (
    collect_source_node,
    fetch_logs_node,
    filter_logs_node,
    format_report_node,
    llm_reasoning_node,
    publish_comment_node,
    redact_and_sandbox_node,
    vector_lookup_node,
)
from graph.state import AgentState

logger = logging.getLogger("central_agent.workflow")

# Attempt LangGraph compilation
_compiled_app = None

try:
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("fetch_logs", fetch_logs_node)
    workflow.add_node("filter_logs", filter_logs_node)
    workflow.add_node("redact_and_sandbox", redact_and_sandbox_node)
    workflow.add_node("collect_source", collect_source_node)
    workflow.add_node("vector_lookup", vector_lookup_node)
    workflow.add_node("llm_reasoning", llm_reasoning_node)
    workflow.add_node("format_report", format_report_node)
    workflow.add_node("publish_comment", publish_comment_node)

    # Define Linear Flow
    workflow.set_entry_point("fetch_logs")
    workflow.add_edge("fetch_logs", "filter_logs")
    workflow.add_edge("filter_logs", "redact_and_sandbox")
    workflow.add_edge("redact_and_sandbox", "collect_source")
    workflow.add_edge("collect_source", "vector_lookup")
    workflow.add_edge("vector_lookup", "llm_reasoning")
    workflow.add_edge("llm_reasoning", "format_report")
    workflow.add_edge("format_report", "publish_comment")
    workflow.add_edge("publish_comment", END)

    _compiled_app = workflow.compile()
    logger.info("Successfully compiled LangGraph CI/CD triage workflow.")

except ImportError:
    logger.warning("LangGraph not installed. Will use modular sequential runner.")
    _compiled_app = None


def run_pipeline(initial_state: AgentState) -> AgentState:
    """Runs the triage pipeline either via LangGraph or modular sequential execution."""
    current_state: AgentState = dict(initial_state)  # type: ignore

    # Fill default fields if missing
    defaults = {
        "raw_logs": "",
        "cleaned_log": "",
        "extracted_evidence": [],
        "source_files": [],
        "untrusted_log_xml": "",
        "untrusted_source_xml": "",
        "relevant_runbooks": [],
        "past_fixes": [],
        "analysis": None,
        "markdown_report": "",
        "status": "started",
        "error_message": None,
    }
    for k, v in defaults.items():
        if k not in current_state:
            current_state[k] = v  # type: ignore

    if _compiled_app is not None:
        logger.info("Executing via LangGraph StateGraph engine...")
        result = _compiled_app.invoke(current_state)
        return result  # type: ignore

    logger.info("Executing via sequential node runner...")
    steps = [
        ("fetch_logs", fetch_logs_node),
        ("filter_logs", filter_logs_node),
        ("redact_and_sandbox", redact_and_sandbox_node),
        ("collect_source", collect_source_node),
        ("vector_lookup", vector_lookup_node),
        ("llm_reasoning", llm_reasoning_node),
        ("format_report", format_report_node),
        ("publish_comment", publish_comment_node),
    ]

    for step_name, node_func in steps:
        logger.debug("Running step: %s", step_name)
        updates = node_func(current_state)
        current_state.update(updates)

    return current_state
