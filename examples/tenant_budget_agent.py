"""tenant_budget_agent.py — pre-tool-call authorization with CyclesToolGate.

A LangChain agent gated by Cycles. Two scenarios:

  1. Tenant ``acme`` has budget; the email tool is allowed; agent runs to completion.
  2. Tenant ``trial`` is over its cap; the email tool is denied; the agent recovers
     gracefully via a ``ToolMessage`` and reports the denial back to the user.

Run::

    pip install langchain-runcycles langchain langchain-openai
    export CYCLES_BASE_URL=http://localhost:7878
    export CYCLES_API_KEY=...
    export OPENAI_API_KEY=...
    python tenant_budget_agent.py
"""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_runcycles import CyclesToolGate
from runcycles import Action, CyclesClient, CyclesConfig, Subject


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient with the provided body."""
    return f"Sent email to {to}"


@tool
def search_docs(query: str) -> str:
    """Search the company knowledge base."""
    return f"Top result for {query!r}: see runcycles.io"


def build_client() -> CyclesClient:
    return CyclesClient(
        CyclesConfig(
            base_url=os.environ.get("CYCLES_BASE_URL", "http://localhost:7878"),
            api_key=os.environ["CYCLES_API_KEY"],
        )
    )


def build_agent(tenant: str) -> object:
    client = build_client()
    gate = CyclesToolGate(
        client,
        subject=Subject(tenant=tenant, agent="tenant-budget-demo"),
        action={
            "send_email": Action(kind="tool.call", name="send_email"),
            "search_docs": Action(kind="tool.call", name="search_docs"),
        },
        mode="decide",
        denial_message="Cycles denied {tool} for tenant — reason: {reason}",
    )
    return create_agent(
        model=os.environ.get("MODEL", "claude-sonnet-4-6"),
        tools=[send_email, search_docs],
        middleware=[gate],
    )


def main() -> None:
    for tenant in ("acme", "trial"):
        print(f"\n=== Run for tenant={tenant} ===")
        agent = build_agent(tenant)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Email alice@example.com to confirm the meeting."}]}
        )
        for message in result["messages"]:
            print(f"  [{type(message).__name__}] {getattr(message, 'content', '')[:200]}")


if __name__ == "__main__":
    main()
