[![PyPI](https://img.shields.io/pypi/v/langchain-runcycles)](https://pypi.org/project/langchain-runcycles/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/langchain-runcycles)](https://pypi.org/project/langchain-runcycles/)
[![CI](https://github.com/runcycles/langchain-runcycles/actions/workflows/ci.yml/badge.svg)](https://github.com/runcycles/langchain-runcycles/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/runcycles/langchain-runcycles/actions)

# Cycles for LangChain — AI agent middleware for budget and action authority

**LangChain middleware for pre-execution budget authority over model calls, tool calls, and runaway agent loops in `create_agent` workflows.** Provider-neutral: works with any LangChain 1.x agent regardless of model provider, as long as actions flow through LangChain middleware/tool execution.

Built on LangChain's [`AgentMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware/) API:

- **`wrap_model_call`** — pre-model-call authorization plus optional reserve/commit/release lifecycle around each LLM invocation (v0.1.5+)
- **`wrap_tool_call`** — tool-call authorization plus optional reserve/commit/release lifecycle around each tool execution
- **`before_model`** (with `@hook_config(can_jump_to=["end"])`) — fan-out caps and external policy halts before another model turn

Per-call actual-cost extraction is available on `CyclesModelGate` via the `cost_fn` parameter (v0.2.0+): supply a `Callable[[ModelResponse], Amount]` and commits debit at actual provider-reported token usage instead of the configured `estimate`. `langchain_runcycles.extractors` ships `openai_cost` and `anthropic_cost` factories parameterized by per-million-token pricing. For non-agent LangChain code (bare chains, RAG runnables), the `BaseCallbackHandler` recipe in [`cycles-client-python/examples/langchain_integration.py`](https://github.com/runcycles/cycles-client-python/blob/main/examples/langchain_integration.py) remains the right tool.

Install via `pip install langchain-runcycles`.

## What's in the box

- **`CyclesModelGate`** (v0.1.5+) — runs before every model call. Authorizes via `client.decide()` and/or reserves budget. Returns a `ModelResponse` carrying the denial reason on deny so the agent terminates naturally.
- **`CyclesToolGate`** — runs before every tool call. Authorizes via `client.decide()` and/or reserves budget via `client.create_reservation()`. Returns a `ToolMessage` on denial so the model can recover gracefully.
- **`CyclesFanOutGate`** — runs before every model turn. Halts the agent (with `jump_to: "end"`) when a turn cap is hit or when an external policy says to stop. Useful for runaway-loop protection and per-tenant burst caps.

All three work with sync or async LangChain agents and the sync (`CyclesClient`) or async (`AsyncCyclesClient`) Cycles client. Compose them in a single `middleware=[...]` list — typical order is `[CyclesFanOutGate, CyclesModelGate, CyclesToolGate]` so fan-out caps trigger before model spend before tool side effects.

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

### `CyclesModelGate` (v0.1.5+)

Gates each model call. Same three modes as `CyclesToolGate`. On denial in `decide` mode, returns a `ModelResponse` whose `AIMessage` carries the denial reason — the agent terminates naturally because the AIMessage has no `tool_calls`.

```python
from langchain_runcycles import CyclesModelGate

model_gate = CyclesModelGate(
    client,
    subject=Subject(tenant="acme", agent="researcher"),
    action=Action(kind="llm.completion", name="gpt-4o"),
    mode="reserve",
    estimate=Amount(unit=Unit.USD_MICROCENTS, amount=2_000_000),  # $0.02 per call
)
```

> Add `cost_fn=openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00)` (or `anthropic_cost(...)`, or a custom `Callable[[ModelResponse], Amount]`) to commit at actual reported token usage instead of `estimate` (v0.2.0+). See the "Actual-cost extraction on `CyclesModelGate`" section below for the full pattern.

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

# Callable — receives the LangChain ToolCallRequest. Pull the run id from
# wherever your runtime carries it: request state, a contextvar, your own
# middleware, etc.
def my_run_id(request):
    return request.state["run_id"]

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

### Actual-cost extraction on `CyclesModelGate` (v0.2.0+)

Reserve-mode model calls commit at the configured `estimate` by default. Pass a `cost_fn` to commit at actual provider-reported token usage instead:

```python
from langchain_runcycles import CyclesModelGate
from langchain_runcycles.extractors import anthropic_cost, openai_cost
from runcycles import Action, Amount, Subject, Unit

# OpenAI gpt-4o pricing (2026-05): $2.50/M input, $10.00/M output
gate = CyclesModelGate(
    client,
    subject=Subject(tenant="acme"),
    action=Action(kind="llm.completion", name="gpt-4o"),
    mode="reserve",
    estimate=Amount(unit=Unit.USD_MICROCENTS, amount=2_000_000),  # worst-case headroom
    cost_fn=openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00),
)

# Anthropic claude-sonnet-4-6 pricing (2026-05): $3.00/M input, $15.00/M output
gate = CyclesModelGate(
    client,
    subject=Subject(tenant="acme"),
    action=Action(kind="llm.completion", name="claude-sonnet-4-6"),
    mode="reserve",
    estimate=Amount(unit=Unit.USD_MICROCENTS, amount=2_500_000),
    cost_fn=anthropic_cost(input_per_million_usd=3.00, output_per_million_usd=15.00),
)
```

Both factories read `AIMessage.usage_metadata` (LangChain's normalized usage shape, populated by `langchain-openai` and `langchain-anthropic`) and return an `Amount` in `USD_MICROCENTS`. Pricing arguments are keyword-only so they can't be swapped accidentally.

You can also pass a custom `cost_fn: Callable[[ModelResponse], Amount]` — the middleware just calls it after the wrapped handler returns and uses the result for the commit. **If your callable raises, the gate logs a warning and falls back to `estimate`** — a costing bug never erases the model result.

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

## Known limitations (v0.2)

- **`CyclesToolGate` reserve mode commits at the configured `estimate`, not actual usage.** Per-tool actual-cost instrumentation (analogous to `CyclesModelGate.cost_fn`) is still on the roadmap; set `estimate` to the worst-case spend per call you're willing to debit, or use `mode="decide"` for policy gating without budget movement.
- **Streaming verification deferred.** `CyclesModelGate.wrap_model_call` runs once per model turn and the commit fires after the handler returns; we haven't yet shipped explicit tests against `astream`/`astream_events` to confirm the commit waits for stream completion. Tracked on the v0.2.x roadmap (issue #13).
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
- `langchain-core >= 1.0, < 2.0`

## License

Apache-2.0. See [LICENSE](LICENSE).
