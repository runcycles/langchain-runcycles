"""multi_agent_fanout.py — composing the full agent governance triad.

A research agent that fans out across subagents to investigate a topic, then
drafts an email a human must approve before sending. The pattern stress-tests
all three Cycles middleware classes plus LangChain's built-in
``HumanInTheLoopMiddleware`` in a realistic multi-tenant shape:

  * :class:`CyclesFanOutGate` caps the number of model turns per run,
    halting runaway loops or unbounded sub-agent fan-out before they consume
    further model spend.
  * :class:`CyclesModelGate` (v0.1.5+) gates each LLM call. With a
    ``cost_fn`` from :mod:`langchain_runcycles.extractors` (v0.2.0+), commits
    debit at the model's actual reported token usage — not a worst-case
    estimate — so per-tenant accounting matches what the provider billed you.
  * :class:`CyclesToolGate` (in ``decide+reserve`` mode) gates every tool
    call, reserving budget per call and committing on success — so a
    half-finished run leaves no orphan reservations.
  * :class:`HumanInTheLoopMiddleware` inserts a confirmation step before
    ``send_email``. Cycles authorizes the policy and budget at tool execution
    time; HITL is the human approval layer. The two compose cleanly: a human
    rejection prevents the tool from running, and Cycles can still deny the
    approved tool call before the side effect happens.

This example deliberately matches LangChain's stated co-marketing bar —
multi-agent, HITL, multi-tenant, non-trivial failure modes (per-tenant
budget exhaustion mid-run, individual tool denials that don't crash the
whole run). See ``examples/multi_agent_fanout_writeup.md`` for the
problem-framed pattern walkthrough.

Run::

    pip install langchain-runcycles langchain langchain-anthropic
    export CYCLES_BASE_URL=http://localhost:7878
    export CYCLES_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python multi_agent_fanout.py
"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, HumanInTheLoopMiddleware
from langchain_core.tools import tool
from runcycles import Action, Amount, CyclesClient, CyclesConfig, Subject, Unit

from langchain_runcycles import CyclesFanOutGate, CyclesModelGate, CyclesToolGate
from langchain_runcycles.extractors import anthropic_cost

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


class ResearchAgentState(AgentState, total=False):
    """Agent state with per-run tenant config preserved for middleware."""

    config: dict[str, Any]


TOOL_ACTION_MAP = {
    "search": Action(kind="tool.call", name="search"),
    "summarize": Action(kind="tool.call", name="summarize"),
    "send_email": Action(kind="tool.call", name="send_email"),
    "spawn_subagent": Action(kind="tool.call", name="spawn_subagent"),
}


def per_tenant_subject(_request: Any, state: Any) -> Subject:
    """Resolve the subject from agent state — supports per-run tenant context.

    Each tenant gets their own budget scope on the Cycles server: a tenant who
    has burned through their daily research budget gets denied at decide()
    without ever reaching the model. The same agent process can serve many
    tenants concurrently because subject resolution happens per-call."""
    tenant: str = "acme"
    if isinstance(state, dict):
        cfg = state.get("config") or {}
        tenant = cfg.get("tenant", tenant)
    return Subject(tenant=tenant, agent="researcher", toolset="research-tools")


def build_agent(model: Any | None = None) -> Any:
    """Build the multi-tenant research agent with the full governance triad."""
    client = CyclesClient(
        CyclesConfig(
            base_url=os.environ.get("CYCLES_BASE_URL", "http://localhost:7878"),
            api_key=os.environ["CYCLES_API_KEY"],
        )
    )

    # Gate 1: per-turn fan-out cap. Cheapest gate — counts AIMessages locally and
    # only calls decide() when a Cycles client is supplied. Halts before another
    # model turn so runaway agents can't burn the model budget below.
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

    # Gate 2: per-model-call authorization + actual-cost commit. The model gate
    # reserves the worst-case estimate before the LLM call, then commits at the
    # provider-reported actual token cost via cost_fn — so budget accounting
    # matches your Anthropic invoice line-by-line.
    model_gate = CyclesModelGate(
        client,
        subject=per_tenant_subject,
        action=Action(kind="llm.completion", name="claude-sonnet-4-6"),
        mode="decide+reserve",
        estimate=Amount(unit=Unit.USD_MICROCENTS, amount=2_500_000),  # $0.025 headroom
        cost_fn=anthropic_cost(
            # claude-sonnet-4-6 pricing (2026-05): $3.00/M input, $15.00/M output.
            input_per_million_usd=3.00,
            output_per_million_usd=15.00,
        ),
    )

    # Gate 3: per-tool authorization. decide+reserve mode runs decide() first
    # (cheap policy check) and only reserves budget if the tool is allowed.
    tool_gate = CyclesToolGate(
        client,
        subject=per_tenant_subject,
        action=TOOL_ACTION_MAP,
        mode="decide+reserve",
    )

    # HITL gate: human approval on the side-effecting tool. LangChain asks for
    # human review after the model proposes a tool call and before tool execution.
    # If the human approves, CyclesToolGate still authorizes/reserves the tool
    # immediately before the side effect can happen.
    hitl = HumanInTheLoopMiddleware(
        interrupt_on={"send_email": True},
    )

    # Composition order: fan-out (cheapest) → model (before LLM spend) →
    # HITL (human approval after the model proposes a tool) → tool gate
    # (final machine authorization before side effects).
    return create_agent(
        model=model or os.environ.get("MODEL", "claude-sonnet-4-6"),
        tools=[search, summarize, send_email, spawn_subagent],
        state_schema=ResearchAgentState,
        middleware=[fanout_gate, model_gate, hitl, tool_gate],
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
