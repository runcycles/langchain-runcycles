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

- 57 tests across `tests/test_tool_gate.py`, `tests/test_tool_gate_async.py`, `tests/test_fanout.py`, `tests/test_fanout_async.py`.
- Coverage 98.85% (gate `fail_under = 95` per `pyproject.toml`).
- Both sync (`.invoke()`) and async (`.ainvoke()`) paths exercised.
- Mocking is done at the SDK boundary (`CyclesClient.decide`, etc.) so tests are independent of HTTP transport.

## Known limitations (v0.1)

- **Reserve mode commits at estimate**, not at actual usage. Tool-level cost instrumentation is left to the caller. A future revision may expose a `cost_fn` analogous to `stream_reservation`.
- **No streaming-LLM cost integration yet.** For per-LLM-call streaming budget tracking with token-level commits, use the existing `runcycles.stream_reservation` from inside an LLM-spend handler; future revisions will wire `wrap_model_call` to this directly.
- **Single tenant per middleware instance** unless you supply a `SubjectExtractor` callable. Per-call subject resolution is fully supported via the callable form; only the static-Subject convenience is single-tenant.

## Update protocol

Update this document when:
- A LangChain middleware API change forces a breaking signature change in this package
- A new hook is added (e.g. `wrap_model_call`, `after_model`)
- An SDK method consumed here gains/loses fields that affect the middleware contract
- Async behavior diverges between LangChain runtime versions
