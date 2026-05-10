[![PyPI](https://img.shields.io/pypi/v/langchain-runcycles)](https://pypi.org/project/langchain-runcycles/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/langchain-runcycles)](https://pypi.org/project/langchain-runcycles/)
[![CI](https://github.com/runcycles/langchain-runcycles/actions/workflows/ci.yml/badge.svg)](https://github.com/runcycles/langchain-runcycles/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/runcycles/langchain-runcycles/actions)

# Cycles for LangChain — AI agent middleware for budget and action authority

**LangChain middleware for pre-tool-call authorization, fan-out caps, and per-tenant budget enforcement in `create_agent` workflows.** Provider-neutral: works with any LangChain 1.x agent regardless of model provider, as long as actions flow through LangChain middleware/tool execution.

Built on LangChain's [`AgentMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware/) API:

- **`wrap_tool_call`** — tool-call authorization plus optional reserve/commit/release lifecycle around each tool execution
- **`before_model`** (with `@hook_config(can_jump_to=["end"])`) — fan-out caps and external policy halts before another model turn

Model-call reservation via `wrap_model_call` is on the roadmap but **not implemented in v0.1.x**. For token-level streaming budget tracking today, use `runcycles.stream_reservation` directly inside an LLM-spend handler.

Install via `pip install langchain-runcycles`.

## What's in the box

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

### Idempotency-key namespacing (v0.1.3+)

Cycles idempotency keys default to `{prefix}-{tool_call_id}` — deterministic per tool call so retries land on the same reservation. If your runtime can reuse short tool-call ids across runs (`tc_1`, `tc_2`, ...), set `idempotency_namespace` on the middleware to scope keys by run / workflow / tenant. Keys then become `{prefix}-{namespace}-{tool_call_id}`.

```python
# Static — same namespace every call
gate = CyclesToolGate(
    client,
    subject=Subject(tenant="acme"),
    action=Action(kind="tool.call", name="send_email"),
    idempotency_namespace="run_2026_05_10_abc",
)

# Callable — receives the LangChain ToolCallRequest. Pull the namespace from
# wherever your runtime exposes the run id (a contextvar, your own middleware,
# request metadata, etc.). The exact accessor depends on your LangChain version.
def my_run_id(_request):
    # In production: return your_runtime.current_run_id()
    return current_run_id_contextvar.get("default")

gate = CyclesToolGate(
    client,
    subject=Subject(tenant="acme"),
    action=Action(kind="tool.call", name="send_email"),
    idempotency_namespace=my_run_id,
)
```

`CyclesFanOutGate.idempotency_namespace` is the same shape; the callable receives the agent `state` instead of the tool-call request. Without `idempotency_namespace`, keys keep the v0.1.2 shape exactly — no behavior change.

**Per-call opt-out**: a callable that returns `None` (or empty string) for a particular call disables namespacing *for that call only*, producing the v0.1.2 shape `{prefix}-{tool_call_id}`. Useful when some calls should be globally scoped (admin / system tools) while others get run-scoped namespacing — branch on the request and return `None` from the unscoped path.

**Errors in the callable propagate**: if your callable raises, the exception surfaces from `wrap_tool_call` / `before_model` to the agent. This is intentional — fail-fast on a misconfigured callable rather than silently producing keys with no namespace. Wrap in try/except inside the callable if you want a fallback.

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

### Settlement (commit) failures

In `"reserve"` and `"decide+reserve"` modes, the tool runs first, then the reservation is committed. If the commit call itself fails (network blip, server overload, etc.), the tool already ran — its side effect is real. You have two reasonable options, controlled by `settlement_error_policy`:

| Policy | Behavior | When to choose |
|---|---|---|
| `"raise"` (default) | Propagate the commit exception to the agent. The tool's return value is lost. | Strict governance — no tool-level cost can go unaccounted. |
| `"log"` | Log a warning, return the tool result anyway. The reservation will eventually expire via TTL. | UX-first — keep the agent moving, accept best-effort accounting. |

```python
gate = CyclesToolGate(
    client,
    subject=...,
    action=...,
    mode="reserve",
    settlement_error_policy="log",   # opt out of strict default
)
```

**Trade-off worth understanding:** `"raise"` surfaces the commit failure as a tool exception, so a LangChain agent may retry — at which point the tool's side effect (e.g. an email send, a payment, a CRM write) **repeats**. Choose `"log"` if your tool's side effects are not safely idempotent on retry.

This only affects commit (success-path settlement); release on tool failure always logs and continues so the original tool exception wins.

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

## Known limitations (v0.1)

- **Reserve mode commits at the configured `estimate`, not actual usage.** `mode="reserve"` and `mode="decide+reserve"` reserve the estimate, run the tool, then commit *the same amount* on success. Per-tool actual-cost instrumentation (analogous to `runcycles.stream_reservation`'s `cost_fn`) is on the roadmap. Until then, set `estimate` to the worst-case spend per call you're willing to debit, or use `mode="decide"` if you only want policy gating without budget movement.
- **No model-call middleware yet.** `wrap_model_call` is on the roadmap (planned for v0.2 as `CyclesModelGate`); v0.1.x covers tool-call gating and fan-out caps only. For LLM-spend tracking today, use `runcycles.stream_reservation` directly inside an LLM-spend handler.
- **Per-call subject only via the extractor form.** Static `Subject` pins one tenant per middleware instance. For per-tenant/per-agent routing in a multi-tenant deployment, supply a `SubjectExtractor` callable.
- **Idempotency keys are deterministic only when `tool_call_id` is present.** Keys take the shape `{prefix}-{tool_call_id}` so retries land on the same Cycles reservation. If the upstream omits `tool_call_id`, the middleware synthesizes a fresh `missing-<hex>` id (and logs a warning) — that path is non-deterministic across retries because the synthesis itself is random. Conformant LangChain runtimes always supply `id`.

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
