# langchain-runcycles — Middleware API Conformance Audit

**Date:** 2026-05-10
**Package:** `langchain-runcycles` v0.1.0
**LangChain target:** `langchain >= 1.0, < 2.0` (tested against `langchain==1.2.18`, `langchain-core==1.3.3`, `langgraph==1.1.10`)
**Cycles SDK target:** `runcycles >= 0.4.1` (tested against `runcycles==0.4.1`, Python 3.10+)
**Server audit:** Cycles protocol conformance is owned by [`cycles-client-python/AUDIT.md`](https://github.com/runcycles/cycles-client-python/blob/main/AUDIT.md). This document audits this package's contract with the LangChain agent middleware API only.

---

## Summary

| Category | Pass | Issues |
|----------|------|--------|
| Middleware base class & hooks | 4/4 | 0 |
| Hook return shapes | 3/3 | 0 |
| Sync/async parity | 2/2 | 0 |
| SDK methods consumed | 5/5 | 0 |
| Idempotency-key generation | — | 0 |
| Reservation lifecycle (reserve → commit/release) | — | 0 |
| Test coverage gate | ≥95% | 0 (98.85%) |

**Overall: middleware contract is in conformance with the LangChain 1.x API as documented at <https://docs.langchain.com/oss/python/langchain/middleware/custom>.**

---

## Audit Scope

Compared the following across LangChain documentation and this package's source:

- `AgentMiddleware` subclassing and hook overrides
- `wrap_tool_call`, `before_model` (sync); `awrap_tool_call`, `abefore_model` (async)
- `@hook_config(can_jump_to=["end"])` usage on fan-out halt
- `ToolMessage` shape on denial (`tool_call_id`, `content`)
- `jump_to: "end"` halt return shape
- `AsyncCyclesClient` parity with `CyclesClient` for every consumed SDK method

## Hooks used

| Hook | File:Line | Notes |
|---|---|---|
| `wrap_tool_call(self, request, handler)` | `langchain_runcycles/tool_gate.py:80` | Sync. Reads `request.tool_call['name'/'args'/'id']` and `request.state` (best-effort). Returns `ToolMessage` on deny, else `handler(request)`. |
| `awrap_tool_call(self, request, handler)` | `langchain_runcycles/tool_gate.py:160` | Async. Awaits the SDK; awaits `handler(request)` if it returns a coroutine. |
| `before_model(self, state, runtime)` | `langchain_runcycles/fanout.py:81` | Sync. Decorated with `@hook_config(can_jump_to=["end"])`. Returns `None` when allowed, halt-dict otherwise. |
| `abefore_model(self, state, runtime)` | `langchain_runcycles/fanout.py:113` | Async. Same contract. |

## Halt-return shape

```python
{"messages": [AIMessage(content=...)], "jump_to": "end"}
```

`jump_to` target is `"end"`, declared in `@hook_config(can_jump_to=["end"])` so the LangChain runtime accepts the halt without raising.

## ToolMessage shape

```python
ToolMessage(content=<denial-string>, tool_call_id=<request.tool_call['id']>)
```

`tool_call_id` is required by LangChain; we pass through the original tool-call id from `request.tool_call`. If the request lacks an id (defensive case), an empty string is used and the LangChain runtime surfaces the denial without correlation.

## SDK methods consumed

| Method | Used in | Mode |
|---|---|---|
| `client.decide(DecisionRequest)` | `tool_gate.py` (decide / decide+reserve), `fanout.py` (when client provided) | sync + async |
| `client.create_reservation(ReservationCreateRequest)` | `tool_gate.py` (reserve / decide+reserve) | sync + async |
| `client.commit_reservation(reservation_id, CommitRequest)` | `tool_gate.py` (reserve / decide+reserve, success path) | sync + async |
| `client.release_reservation(reservation_id, ReleaseRequest)` | `tool_gate.py` (reserve / decide+reserve, exception path) | sync + async |
| `CyclesResponse.{is_success, body, get_body_attribute, get_error_response}` | `_internal.py` | n/a |

Type model imports from `runcycles`:
- `Action`, `Amount`, `AsyncCyclesClient`, `CommitRequest`, `CyclesClient`, `Decision`, `DecisionRequest`, `DecisionResponse`, `ReleaseRequest`, `ReservationCreateRequest`, `Subject`, `Unit`, `CyclesResponse`.

## Idempotency keys

Each Cycles request gets a unique idempotency key derived from a short prefix plus a UUID hex suffix. When a tool-call id is available, it is included in the key for traceability:

| Operation | Key shape |
|---|---|
| `decide` | `decide-<tool_call_id>-<8-hex>` or `decide-<32-hex>` |
| `create_reservation` | `res-<tool_call_id>-<8-hex>` or `res-<32-hex>` |
| `commit_reservation` | `commit-<reservation-key>` (key from create) |
| `release_reservation` | `release-<reservation-key>` (key from create) |
| `decide` (fanout) | `fanout-decide-<32-hex>` |

This matches the parent SDK's idempotency-key conventions.

## Reservation lifecycle

`tool_gate.py` paths in `reserve` / `decide+reserve` mode:

1. Pre-call: `create_reservation` → if not success or no `reservation_id`, return `ToolMessage` denial.
2. Run handler.
3. Success: `commit_reservation` (commits at the configured `estimate`; tool-level actual-cost instrumentation is left to the caller for v0.1).
4. Exception: `release_reservation`, then re-raise.

Commit/release failures are logged and swallowed (the tool result must not be masked by a Cycles bookkeeping error).

## Test coverage

- 74 tests across:
  - `tests/test_tool_gate.py`, `tests/test_tool_gate_async.py` — sync + async tool-gate paths (including settlement_error_policy raise/log + idempotency-key determinism)
  - `tests/test_fanout.py`, `tests/test_fanout_async.py` — sync + async fan-out paths
  - `tests/test_examples.py` — import smoke for bundled examples
  - `tests/integration/test_live_agent.py` — `create_agent` construction with our middleware against a `FakeMessagesListChatModel`, verifying the AgentMiddleware contract is satisfied at runtime
- Coverage ≥99% (gate `fail_under = 95` per `pyproject.toml`).
- Both sync (`.invoke()`) and async (`.ainvoke()`) paths exercised.
- Mocking is done at the SDK boundary (`CyclesClient.decide`, etc.) so tests are independent of HTTP transport.
- Idempotency-key determinism (`<prefix>-<tool_call_id>`, no random suffix) and reserve-mode commit amount (`actual=estimate`) are explicitly asserted to prevent silent contract drift.

## Known limitations (v0.1.x)

- **Reserve mode commits at estimate**, not at actual usage. Tool-level cost instrumentation is left to the caller. A future revision may expose a `cost_fn` analogous to `stream_reservation`. Locked down by `tests/test_tool_gate.py::test_commit_called_with_configured_estimate`.
- **No model-call middleware yet.** `CyclesModelGate` (using `wrap_model_call`) is planned for v0.2; v0.1.x covers tool-call gating and fan-out caps only. For LLM-spend tracking today, use `runcycles.stream_reservation` directly inside an LLM-spend handler.
- **Single tenant per middleware instance** unless you supply a `SubjectExtractor` callable. Per-call subject resolution is fully supported via the callable form; only the static-Subject convenience is single-tenant.
- **Synthetic `tool_call_id` when missing.** A `ToolCallRequest` with no `id` field has its denial `ToolMessage` correlated via a fabricated `missing-<12-hex>` id, with a warning logged at `langchain_runcycles._internal`. Because the synthesis is fresh per call, the resulting idempotency key on this fallback path is *not* retry-stable. Conformant LangChain runtimes always supply `id`. Locked down by `tests/test_tool_gate.py::test_synthetic_tool_call_id_when_missing`.
- **Fan-out gate rejects per-tool action mappings.** `CyclesFanOutGate` gates *model turns*, not tool calls; a per-tool-name `Mapping` for `action` is meaningless there and is rejected at construction with `TypeError`. Locked down by `tests/test_fanout.py::test_fanout_rejects_mapping_action`.

## Settlement error policy (v0.1.2+)

The `commit_reservation` call happens *after* the gated tool already ran, so a commit failure has two reasonable resolutions and `CyclesToolGate` exposes them as `settlement_error_policy`:

| Policy | Behavior |
|---|---|
| `"raise"` (default) | Surface the commit exception. Tool result is lost; caller reconciles. Strict-governance default — no tool-level cost goes unaccounted. |
| `"log"` | Log a warning, return the tool result. Reservation expires via TTL. Best-effort accounting; preferred when UX continuity matters more than per-call settlement guarantees. |

The release path (on tool-side exception) always logs and continues so the original tool exception wins; settlement_error_policy applies only to the success-path commit.

Locked down by `test_settlement_raise_default_propagates_commit_failure`, `test_settlement_log_swallows_commit_failure`, and async siblings.

## Idempotency-key determinism (v0.1.2+)

Idempotency keys take the shape `{prefix}-{tool_call_id}` with no random component when the upstream supplies a tool call id. This is a behavior change from v0.1.0/v0.1.1, which appended a random 8-hex suffix.

The deterministic shape is the correctness story: a duplicate dispatch (durable workflow replay, middleware retry, process recovery) lands on the same Cycles reservation rather than creating a second one. UUID fallback is used only when `coerce_tool_call_id` had to synthesize a missing id — that path is non-deterministic and a known limitation (see above).

## Update protocol

Update this document when:
- A LangChain middleware API change forces a breaking signature change in this package
- A new hook is added (e.g. `wrap_model_call`, `after_model`)
- An SDK method consumed here gains/loses fields that affect the middleware contract
- Async behavior diverges between LangChain runtime versions
