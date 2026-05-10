"""multi_agent_fanout.py — multi-agent / HITL flow with budget enforcement.

A research agent that spawns subagents to investigate a topic. Each subagent
makes one or two tool calls. The pattern is interesting because it stress-tests
both middleware in a realistic shape:

  * ``CyclesFanOutGate`` caps the number of model turns per run, preventing
    runaway loops or unbounded sub-agent fan-out.
  * ``CyclesToolGate`` (in ``decide+reserve`` mode) gates every tool call,
    reserving budget per call and committing on success — so a half-finished
    run leaves no orphan reservations.
  * A ``HumanInTheLoopMiddleware`` from LangChain inserts a confirmation step
    before any ``send_email`` call. Cycles authorizes the call separately;
    HITL is the human approval layer.

This example is the kind of complex end-to-end LangGraph workflow that
LangChain's co-marketing guidelines call out as promotion-worthy: long
horizon, multi-agent, HITL, with non-trivial failure modes (budget runs out
mid-run; one subagent's denial does not crash the whole run).

Run::

    pip install langchain-runcycles langchain langchain-openai
    export CYCLES_BASE_URL=http://localhost:7878
    export CYCLES_API_KEY=...
    export OPENAI_API_KEY=...
    python multi_agent_fanout.py
"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool

from langchain_runcycles import CyclesFanOutGate, CyclesToolGate
from runcycles import Action, CyclesClient, CyclesConfig, Subject


# --- tools ---


@tool
def search(topic: str) -> str:
    """Search the public web for a topic."""
    return f"Search results for {topic!r}: ..."


@tool
def summarize(text: str) -> str:
    """Summarize a block of text."""
    return f"Summary: {text[:60]}..."


@tool
def send_email(to: str, body: str) -> str:
    """Send an email — gated by Cycles policy AND human approval."""
    return f"Sent to {to}"


@tool
def spawn_subagent(query: str) -> str:
    """Spawn a subagent to investigate a sub-question. Returns the subagent's findings."""
    # In a real implementation this would invoke a child agent. Stubbed here for clarity.
    return f"Subagent finding for {query!r}: ..."


# --- subject + action mapping ---

ACTION_MAP = {
    "search": Action(kind="tool.call", name="search"),
    "summarize": Action(kind="tool.call", name="summarize"),
    "send_email": Action(kind="tool.call", name="send_email"),
    "spawn_subagent": Action(kind="tool.call", name="spawn_subagent"),
}


def per_tenant_subject(_request: Any, state: Any) -> Subject:
    """Resolve the subject from agent state — supports per-run tenant context."""
    tenant: str = "acme"
    if isinstance(state, dict):
        cfg = state.get("config") or {}
        tenant = cfg.get("tenant", tenant)
    return Subject(tenant=tenant, agent="researcher", toolset="research-tools")


def build_agent() -> object:
    client = CyclesClient(
        CyclesConfig(
            base_url=os.environ.get("CYCLES_BASE_URL", "http://localhost:7878"),
            api_key=os.environ["CYCLES_API_KEY"],
        )
    )

    tool_gate = CyclesToolGate(
        client,
        subject=per_tenant_subject,
        action=ACTION_MAP,
        mode="decide+reserve",
    )

    fanout_gate = CyclesFanOutGate(
        max_turns=20,
        client=client,
        subject=per_tenant_subject,
        action=Action(kind="model.turn", name="research"),
        cap_message=(
            "Fan-out cap reached at {turns} turns; halting to prevent runaway. "
            "Increase ``max_turns`` or split the task."
        ),
    )

    hitl = HumanInTheLoopMiddleware(
        interrupt_on={"send_email": True},
    )

    return create_agent(
        model=os.environ.get("MODEL", "claude-sonnet-4-6"),
        tools=[search, summarize, send_email, spawn_subagent],
        middleware=[fanout_gate, tool_gate, hitl],
    )


def main() -> None:
    agent = build_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Research the LangChain agent middleware API. Spawn subagents for "
                        "(1) the new wrap_tool_call hook and (2) the before_model halt "
                        "semantics. Summarize the findings, then email me at "
                        "me@example.com with the summary."
                    ),
                }
            ],
            "config": {"tenant": "acme"},
        }
    )
    for message in result["messages"]:
        print(f"  [{type(message).__name__}] {getattr(message, 'content', '')[:200]}")


if __name__ == "__main__":
    main()
