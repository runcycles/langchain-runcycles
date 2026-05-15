"""tool_cost_fn.py — actual-cost commits for CyclesToolGate.

`CyclesToolGate.cost_fn` receives both the LangChain ToolCallRequest and the
wrapped tool result. That makes it useful for router-style pricing: branch on
the tool name and read either request args or provider-returned metadata.

Run::

    pip install langchain-runcycles langchain langchain-anthropic
    export CYCLES_BASE_URL=http://localhost:7878
    export CYCLES_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python tool_cost_fn.py
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from runcycles import Action, Amount, CyclesClient, CyclesConfig, Subject, Unit

from langchain_runcycles import CyclesToolGate, ToolCostFn


def _tool_content(result: Any) -> Any:
    """Return structured ToolMessage content when the runtime JSON-serializes it."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


@tool
def send_sms(to: str, body: str) -> str:
    """Send an SMS message to a recipient."""
    return f"Sent SMS to {to}"


@tool
def lookup_customer(email: str) -> dict[str, Any]:
    """Look up a customer in a paid enrichment system."""
    return {"email": email, "plan": "pro", "charged_microcents": 12_500}


def tool_cost(request: Any, result: Any) -> Amount:
    """Compute actual tool cost from the request and result shape."""
    tool_name = request.tool_call["name"]
    if tool_name == "send_sms":
        body = request.tool_call.get("args", {}).get("body", "")
        segments = max(1, (len(body) + 159) // 160)
        return Amount(unit=Unit.USD_MICROCENTS, amount=segments * 75_000)

    if tool_name == "lookup_customer":
        content = _tool_content(result)
        if isinstance(content, dict) and isinstance(content.get("charged_microcents"), int):
            return Amount(unit=Unit.USD_MICROCENTS, amount=content["charged_microcents"])
        return Amount(unit=Unit.USD_MICROCENTS, amount=12_500)

    return Amount(unit=Unit.USD_MICROCENTS, amount=0)


def build_client() -> CyclesClient:
    return CyclesClient(
        CyclesConfig(
            base_url=os.environ.get("CYCLES_BASE_URL", "http://localhost:7878"),
            api_key=os.environ["CYCLES_API_KEY"],
        )
    )


def build_agent(cost_fn: ToolCostFn = tool_cost) -> object:
    client = build_client()
    gate = CyclesToolGate(
        client,
        subject=Subject(tenant="acme", agent="tool-cost-demo"),
        action={
            "send_sms": Action(kind="tool.call", name="send_sms"),
            "lookup_customer": Action(kind="tool.call", name="lookup_customer"),
        },
        mode="reserve",
        estimate=Amount(unit=Unit.USD_MICROCENTS, amount=500_000),
        cost_fn=cost_fn,
    )
    return create_agent(
        model=os.environ.get("MODEL", "claude-sonnet-4-6"),
        tools=[send_sms, lookup_customer],
        middleware=[gate],
    )


def main() -> None:
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Look up alice@example.com, then text her a short hello."}]}
    )
    for message in result["messages"]:
        print(f"  [{type(message).__name__}] {getattr(message, 'content', '')[:200]}")


if __name__ == "__main__":
    main()
