# langchain-runcycles

[![PyPI](https://img.shields.io/pypi/v/langchain-runcycles.svg)](https://pypi.org/project/langchain-runcycles/)
[![CI](https://github.com/runcycles/langchain-runcycles/actions/workflows/ci.yml/badge.svg)](https://github.com/runcycles/langchain-runcycles/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

LangChain agent middleware for [Cycles](https://runcycles.io) — pre-tool-call authorization, fan-out caps, and per-tenant budget enforcement for Python agents.

This package exposes two `AgentMiddleware` subclasses that gate agent behavior at runtime against an external Cycles policy/budget service:

- **`CyclesToolGate`** — runs before every tool call. Authorizes via `client.decide()` and/or reserves budget via `client.create_reservation()`. Returns a `ToolMessage` on denial so the model can recover gracefully.
- **`CyclesFanOutGate`** — runs before every model turn. Halts the agent (with `jump_to: "end"`) when a turn cap is hit or when an external policy says to stop. Useful for runaway-loop protection and per-tenant burst caps.

Both work with sync or async LangChain agents and the sync (`CyclesClient`) or async (`AsyncCyclesClient`) Cycles client.

## Installation

```bash
pip install langchain-runcycles
```

Requires Python 3.10+ and `langchain >= 1.0`.

## Quick Start

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_runcycles import CyclesToolGate
from runcycles import Action, CyclesClient, CyclesConfig, Subject

@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"Sent to {to}"

client = CyclesClient(CyclesConfig(base_url="http://localhost:7878", api_key="..."))
gate = CyclesToolGate(
    client,
    subject=Subject(tenant="acme", agent="researcher"),
    action={"send_email": Action(kind="tool.call", name="send_email")},
    mode="decide",
)

agent = create_agent(model="claude-sonnet-4-6", tools=[send_email], middleware=[gate])
agent.invoke({"messages": [{"role": "user", "content": "Email alice."}]})
```

If `client.decide()` denies the call, `send_email` is never invoked — the model receives a `ToolMessage` with the denial reason and can choose another path.

## Middleware

### `CyclesToolGate`

Gates each tool call. Three modes:

| Mode | What it does |
|---|---|
| `"decide"` | Calls `client.decide()`. Denies the tool call on a non-allow decision. No reservation. |
| `"reserve"` | Creates a reservation, runs the tool, commits on success / releases on exception. |
| `"decide+reserve"` | Authorizes via `decide()`, then reserves+commits. Most strict. |

```python
gate = CyclesToolGate(
    client,
    subject=Subject(tenant="acme", agent="researcher"),
    action={
        "search": Action(kind="tool.call", name="search"),
        "send_email": Action(kind="tool.call", name="send_email"),
    },
    mode="decide+reserve",
)
```

### `CyclesFanOutGate`

Halts the agent when a turn cap or external policy says stop. Optional `client` argument enables remote policy checks on each turn:

```python
from langchain_runcycles import CyclesFanOutGate

fanout = CyclesFanOutGate(
    max_turns=20,
    client=client,                       # optional — for remote policy
    subject=Subject(tenant="acme"),
    action=Action(kind="model.turn", name="research"),
)
```

Pair with `CyclesToolGate` and `HumanInTheLoopMiddleware` for production-grade agent governance.

## Configuration

### Subject

Either a static `Subject` or a callable resolving from request/state:

```python
from runcycles import Subject

# Static
subject = Subject(tenant="acme", agent="bot")

# Per-call extractor (CyclesToolGate: (request, state); CyclesFanOutGate: (state, state))
def per_tenant(request, state):
    return Subject(tenant=state["config"]["tenant"], agent="bot")
```

### Action

Static, mapping (per-tool name), or callable:

```python
from runcycles import Action

# Static
action = Action(kind="tool.call", name="any")

# Per-tool mapping
action = {
    "send_email": Action(kind="tool.call", name="send_email"),
    "search": Action(kind="tool.call", name="search"),
}

# Callable
def derive(request):
    return Action(kind="tool.call", name=request.tool_call["name"])
```

### Denial messages

`denial_message` accepts a format string (placeholders: `{reason}`, `{tool}`, `{decision}`) or a callable receiving the `CyclesResponse`:

```python
gate = CyclesToolGate(
    client,
    subject=...,
    action=...,
    denial_message="Cycles denied {tool}: {reason}",
)
```

## Error handling

- **Denied tool calls** return a `ToolMessage` with the denial content; the underlying handler is never invoked. The agent's model sees the denial as if a tool returned an error and can recover.
- **Reservation failures** in `"reserve"` mode are returned as `ToolMessage` (handler not invoked).
- **Tool exceptions** in `"reserve"` mode trigger an automatic `release_reservation`, then the exception propagates.
- **Async/sync mismatch** raises `TypeError` — pair `CyclesClient` with `.invoke()` and `AsyncCyclesClient` with `.ainvoke()`.

## Async support

Async middleware variants run automatically when the LangChain agent is invoked with `.ainvoke()`. Pass an `AsyncCyclesClient`:

```python
from runcycles import AsyncCyclesClient

async_client = AsyncCyclesClient(CyclesConfig(...))
gate = CyclesToolGate(async_client, subject=..., action=..., mode="decide")

agent = create_agent(model="...", tools=[...], middleware=[gate])
await agent.ainvoke({"messages": [...]})
```

## Examples

- [`examples/tenant_budget_agent.py`](examples/tenant_budget_agent.py) — single-tenant budget gate with risky-tool denial recovery.
- [`examples/multi_agent_fanout.py`](examples/multi_agent_fanout.py) — multi-agent / HITL flow with `CyclesToolGate` + `CyclesFanOutGate` + `HumanInTheLoopMiddleware`.

## Development

```bash
pip install -e ".[dev]"
pytest                          # all tests
pytest --cov=langchain_runcycles  # with coverage (gate: ≥95%)
ruff check . && ruff format
mypy langchain_runcycles
```

## Documentation

- LangChain integration page: https://docs.langchain.com/oss/python/integrations/middleware/runcycles (pending PR review)
- Cycles protocol & SDK: https://runcycles.io
- Architecture: see [AUDIT.md](AUDIT.md)

## Requirements

- Python 3.10+
- `runcycles >= 0.4.1`
- `langchain >= 1.0, < 2.0`
- `langchain-core >= 0.3`

## License

Apache-2.0. See [LICENSE](LICENSE).
