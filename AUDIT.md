# langchain-runcycles — Middleware and Recovery Audit

**Date:** 2026-08-06
**Package:** `langchain-runcycles` v0.4.0
**LangChain target:** `langchain >=1.0,<2.0`, `langchain-core >=1.0,<2.0`
**Cycles SDK target:** `runcycles >=0.5.3`
**Python:** 3.10–3.14

This audit covers the package's LangChain middleware contract and the recovery
behavior used by its reserve modes. Wire-level conformance remains owned by
[`cycles-client-python/AUDIT.md`](https://github.com/runcycles/cycles-client-python/blob/main/AUDIT.md).

## Result

| Area | Result |
|---|---|
| LangChain 1.x hook shapes | Pass |
| Sync/async parity | Pass |
| Pre-execution deny behavior | Pass |
| Heartbeat during guarded execution | Pass — delegated to `runcycles` managed reservations |
| Durable known-spend settlement | Pass — journal before first commit, restart replay, event fallback |
| Cost extraction | Pass — normalized usage plus optional cache tiers |
| Test coverage gate | Pass — 187 tests, 99.32% line coverage (`>=95%` required) |

## Middleware hooks

| Class | Hooks | Denial/halt result |
|---|---|---|
| `CyclesModelGate` | `wrap_model_call`, `awrap_model_call` | `ModelResponse(result=[AIMessage(...)])` |
| `CyclesToolGate` | `wrap_tool_call`, `awrap_tool_call` | Correlated `ToolMessage` |
| `CyclesFanOutGate` | `before_model`, `abefore_model` | `{"messages": [...], "jump_to": "end"}` |

`CyclesFanOutGate` uses `@hook_config(can_jump_to=["end"])`. Decide responses
permit both `ALLOW` and `ALLOW_WITH_CAPS`.

## Reserve-mode choreography

`CyclesModelGate` and `CyclesToolGate` use `client.stream_reservation(...)`
for both sync and async reserve modes:

1. Resolve subject, action, estimate, namespace, and reservation key.
2. Create a reservation before invoking the handler. A protocol failure is
   adapted to the gate's normal denial result and the handler is not invoked.
3. Start the SDK heartbeat. Field-bearing servers use the protocol's
   `remaining_ttl_ms` algorithm; older servers use the documented fallback.
4. Invoke the model/tool handler.
5. On handler failure, stop the heartbeat, attempt release, and re-raise the
   original handler failure. Release errors are logged and never mask it.
6. On handler success, resolve actual cost, write a pending settlement record,
   and attempt commit. Schema-valid HTTP 200 is the only success.
7. Transient, authentication, rate-limit, and ambiguous outcomes remain in the
   durable journal. `RESERVATION_EXPIRED` transitions to `/v1/events` recovery.
   Known spend is never released because settlement failed.

The SDK owns the retry state machine and journal format, preventing this
integration from drifting from the shared client recovery contract.

## Settlement error policy

The policy is an observation choice, not a durability choice:

| Policy | Current call | Recovery |
|---|---|---|
| `"raise"` (default) | Raises `CyclesProtocolError` after a synchronous commit failure | Already queued when retryable |
| `"log"` | Logs and returns the handler result | Same durable recovery path |

This avoids the former failure mode where `"log"` meant waiting for TTL expiry
and losing the spend. Callers should still avoid automatically repeating a
non-idempotent handler after the strict policy surfaces a settlement problem.

## Idempotency

- A tool call with an upstream `tool_call_id` uses a deterministic reservation
  key: `{prefix}-{optional_namespace}-{tool_call_id}`.
- Commit and release keys are deterministically derived from a caller-supplied
  reservation key by the SDK.
- Reservation-key combinations over the protocol's 256-character limit use a
  deterministic SHA-256 form while preserving retry stability.
- Missing/malformed tool ids use a logged random synthetic id and are not
  retry-stable across redispatches.
- Model and fan-out hooks have no equivalent upstream stable call id. Their
  keys are random within the optional namespace; documentation does not claim
  retry stability for those hooks.

## Cost extraction

Both built-ins read `AIMessage.usage_metadata`, not provider-specific
`llm_output` fields:

- `openai_cost(prompt_per_million_usd, completion_per_million_usd,
  cached_prompt_per_million_usd=None)`
- `anthropic_cost(input_per_million_usd, output_per_million_usd,
  cache_read_per_million_usd=None, cache_creation_per_million_usd=None,
  cache_creation_5m_per_million_usd=None,
  cache_creation_1h_per_million_usd=None)`

Cache counts come from normalized `input_token_details`, including Anthropic's
5-minute and 1-hour cache-creation breakdown. Omitted tier rates fall back to
the generic cache-creation rate and then the ordinary input rate for backward
compatibility. Rates must be finite and non-negative, detail counts cannot
exceed their totals, and malformed usage raises so the gate can fall back to
its configured estimate. These are caller-supplied flat rates; the package
does not claim to discover live prices.

Tool costs remain user supplied because LangChain has no provider-neutral tool
billing shape. A mismatched unit from either cost callback falls back to the
configured estimate rather than sending a guaranteed `UNIT_MISMATCH` commit.

## Streaming boundary

For completed `agent.astream(...)` / `agent.astream_events(...)` calls,
LangChain aggregates chunk usage into the final `AIMessage` below the
middleware layer. The gate heartbeats throughout execution and commits once
from the final normalized totals.

On cancellation or failure before a final response exists, the middleware has
no finalized normalized usage. It releases the reservation and re-raises.
Provider charges for a partially consumed stream are outside this package's
evidence boundary and require reconciliation from provider billing telemetry.

## Regression coverage

The suite covers:

- sync/async allow, deny, reserve, release, and decide+reserve paths;
- deterministic tool keys and namespace behavior;
- heartbeat-managed reserve paths and settlement `raise`/`log` behavior;
- known-spend journaling through SDK behavior and no release after commit
  rejection;
- model/tool cost callbacks, fallback behavior, mismatched units, normalized
  cache-read/cache-creation pricing, and invalid usage/rates;
- completed streamed aggregation and cancellation behavior;
- real `create_agent` middleware construction with a fake chat model.

Run the evidence locally with:

```bash
pytest --cov=langchain_runcycles --cov-fail-under=95
ruff check .
mypy langchain_runcycles
```

Update this audit whenever a LangChain hook contract changes, the package adds
a recovery path, or the SDK lifecycle guarantee consumed here changes.
