# langchain-runcycles — Middleware API Conformance Audit

**Date:** 2026-05-10
**Package:** `langchain-runcycles` v0.2.3
**LangChain target:** `langchain >= 1.0, < 2.0`, `langchain-core >= 1.0, < 2.0` (tested against `langchain==1.2.18`, `langchain-core==1.3.3`, `langgraph==1.1.10`)
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
| Test coverage gate | ≥95% | 0 (159 tests, 99.62%) |

**Overall: middleware contract is in conformance with the LangChain 1.x API as documented at <https://docs.langchain.com/oss/python/langchain/middleware/custom>.**

---

## Audit Scope

Compared the following across LangChain documentation and this package's source:

- `AgentMiddleware` subclassing and hook overrides for all three middleware classes
- `wrap_model_call`, `wrap_tool_call`, `before_model` (sync); `awrap_model_call`, `awrap_tool_call`, `abefore_model` (async)
- `@hook_config(can_jump_to=["end"])` usage on fan-out halt
- `ToolMessage` shape on tool-gate denial (`tool_call_id`, `content`)
- `ModelResponse(result=[AIMessage(...)])` shape on model-gate denial (terminates agent loop because the AIMessage has no `tool_calls`)
- `jump_to: "end"` halt return shape
- `AsyncCyclesClient` parity with `CyclesClient` for every consumed SDK method
- `settlement_error_policy` / `idempotency_namespace` parity across `CyclesToolGate` and `CyclesModelGate`

## Hooks used

| Hook | File:Line | Notes |
|---|---|---|
| `wrap_tool_call(self, request, handler)` | `langchain_runcycles/tool_gate.py:80` | Sync. Reads `request.tool_call['name'/'args'/'id']` and `request.state` (best-effort). Returns `ToolMessage` on deny, else `handler(request)`. |
| `awrap_tool_call(self, request, handler)` | `langchain_runcycles/tool_gate.py:160` | Async. Awaits the SDK; awaits `handler(request)` if it returns a coroutine. |
| `wrap_model_call(self, request, handler)` | `langchain_runcycles/model_gate.py:108` | Sync (v0.1.5+). Reads `request.state` (best-effort). Returns `ModelResponse(result=[AIMessage(...)])` on deny — agent terminates naturally because the AIMessage has no `tool_calls`. |
| `awrap_model_call(self, request, handler)` | `langchain_runcycles/model_gate.py:188` | Async (v0.1.5+). Awaits the SDK; awaits `handler(request)` if it returns a coroutine. |
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

`tool_call_id` is required by LangChain; we pass through the original id from `request.tool_call`. If the upstream omits `id`, `coerce_tool_call_id` synthesizes `missing-<12-hex>` and logs a warning (see `_internal.py`); the synthesized id is fresh per call, so the resulting idempotency-key path is *not* retry-stable on this fallback.

## SDK methods consumed

| Method | Used in | Mode |
|---|---|---|
| `client.decide(DecisionRequest)` | `tool_gate.py` (decide / decide+reserve), `model_gate.py` (decide / decide+reserve), `fanout.py` (when client provided) | sync + async |
| `client.create_reservation(ReservationCreateRequest)` | `tool_gate.py` + `model_gate.py` (reserve / decide+reserve) | sync + async |
| `client.commit_reservation(reservation_id, CommitRequest)` | `tool_gate.py` + `model_gate.py` (reserve / decide+reserve, success path) | sync + async |
| `client.release_reservation(reservation_id, ReleaseRequest)` | `tool_gate.py` + `model_gate.py` (reserve / decide+reserve, exception path) | sync + async |
| `CyclesResponse.{is_success, body, get_body_attribute, get_error_response}` | `_internal.py` | n/a |

Type model imports from `runcycles`:
- `Action`, `Amount`, `AsyncCyclesClient`, `CommitRequest`, `CyclesClient`, `Decision`, `DecisionRequest`, `DecisionResponse`, `ReleaseRequest`, `ReservationCreateRequest`, `Subject`, `Unit`, `CyclesResponse`.

## Idempotency keys

Idempotency keys are deterministic per `tool_call_id` (v0.1.2+) and may optionally be scoped by a `namespace` (v0.1.3+) to prevent cross-run collisions when frameworks reuse short tool-call ids like `tc_1`.

Key shape, in order of preference:

| `namespace` | `tool_call_id` | Resulting key |
|---|---|---|
| set | set | `{prefix}-{namespace}-{tool_call_id}` (most specific; v0.1.3+) |
| unset | set | `{prefix}-{tool_call_id}` (v0.1.2 shape, retained for back-compat) |
| set | unset | `{prefix}-{namespace}-{32-hex}` (run-scoped, per-call random) |
| unset | unset | `{prefix}-{32-hex}` (last-resort fallback) |

Per-operation prefixes:

| Operation | Prefix |
|---|---|
| `decide` (tool gate) | `decide` |
| `create_reservation` (tool gate) | `res` |
| `commit_reservation` | `commit-{reservation-key}` (composed from create) |
| `release_reservation` | `release-{reservation-key}` (composed from create) |
| `decide` (fanout) | `fanout-decide` |
| `decide` (model gate, v0.1.5+) | `model-decide` |
| `create_reservation` (model gate, v0.1.5+) | `model-res` |

`namespace` is configured via `idempotency_namespace` on `CyclesModelGate` / `CyclesToolGate` / `CyclesFanOutGate` — accepts a static string or a callable. The callable receives the request (model and tool gates) or state (fan-out gate) so users can extract a workflow run id, tenant id, etc. per call.

Locked down by `tests/test_tool_gate.py::test_idempotency_keys_are_deterministic_per_tool_call_id`, `::test_idempotency_key_retry_lands_on_same_key`, `::test_make_idempotency_key_with_namespace_and_suffix`, `::test_idempotency_namespace_as_static_string`, `::test_idempotency_namespace_as_callable`, `::test_namespace_prevents_cross_run_collision`, plus `tests/test_fanout.py::test_fanout_idempotency_namespace_callable_from_state` and async siblings.

## Reservation lifecycle

`tool_gate.py` and `model_gate.py` paths in `reserve` / `decide+reserve` mode (model-gate paths added v0.1.5+):

1. Pre-call: `create_reservation` → if not success or no `reservation_id`, return denial — `ToolMessage` for `CyclesToolGate`, `ModelResponse(result=[AIMessage(...)])` for `CyclesModelGate`.
2. Run handler (the wrapped tool call or model call).
3. Success: `commit_reservation`. For `CyclesModelGate` (v0.2.0+), commits at `cost_fn(result)` if `cost_fn` is supplied (with fallback to `estimate` on extractor error); otherwise commits at `estimate`. For `CyclesToolGate`, always commits at `estimate` (a tool-side `cost_fn` analog is roadmap, not yet shipped).
4. Exception: `release_reservation`, then re-raise.

**Settlement-failure handling** (v0.1.2+ for tool gate, v0.1.5+ for model gate): if the success-path `commit_reservation` itself raises, behavior is governed by `settlement_error_policy` on `CyclesToolGate` and `CyclesModelGate` — default `"raise"` propagates the commit exception so the caller can reconcile (strict governance); opt-in `"log"` swallows the failure and returns the result (best-effort accounting; reservation expires via TTL). The release path on handler-side exception always logs and continues so the original handler exception wins.

## Test coverage

- 159 tests across:
  - `tests/test_tool_gate.py`, `tests/test_tool_gate_async.py` — sync + async tool-gate paths (including settlement_error_policy raise/log, idempotency-key determinism, and v0.1.3 namespace static/callable/no-namespace/cross-run-collision)
  - `tests/test_model_gate.py`, `tests/test_model_gate_async.py` — sync + async model-gate paths (v0.1.5+); decide allow/deny, reserve lifecycle, settlement raise/log, namespace, **plus v0.2.0+ `cost_fn` (applied / None-fallback / exception-fallback / decide-mode-skip)**
  - `tests/test_extractors.py` — `openai_cost` / `anthropic_cost` factories (v0.2.0+); computation, zero-token edge, missing-`usage_metadata` raise, missing token fields raise, negative tokens raise, non-coercible token values raise, empty-`result` raise, fractional-cent rounding, keyword-only-pricing guard
  - `tests/test_model_gate_streaming.py` — streaming-contract verification (v0.2.1+); cost_fn-called-once-per-turn, aggregated-usage-metadata extraction, cancellation-releases-reservation
  - `tests/test_fanout.py`, `tests/test_fanout_async.py` — sync + async fan-out paths (including state-derived idempotency namespace)
  - `tests/test_examples.py` — import smoke for bundled examples
  - `tests/integration/test_live_agent.py` — `create_agent` construction with our middleware against a `FakeMessagesListChatModel`, verifying the AgentMiddleware contract is satisfied at runtime
- Coverage ≥99% (gate `fail_under = 95` per `pyproject.toml`).
- Both sync (`.invoke()`) and async (`.ainvoke()`) paths exercised.
- Mocking is done at the SDK boundary (`CyclesClient.decide`, etc.) so tests are independent of HTTP transport.
- Idempotency-key determinism (`<prefix>-<tool_call_id>`, no random suffix) and reserve-mode commit amount (`actual=estimate`) are explicitly asserted to prevent silent contract drift.

## Per-call cost extraction (v0.2.0+)

`CyclesModelGate` accepts an optional `cost_fn: Callable[[ModelResponse], Amount]`. When set, the gate calls `cost_fn(result)` after the wrapped handler returns and uses the returned `Amount` for `commit_reservation.actual` instead of the configured `estimate`. When unset, behavior is identical to v0.1.x (commit-at-estimate).

| Path | Source | Behavior |
|---|---|---|
| `_resolve_actual(result)` | `model_gate.py:114` | Returns `estimate` if `cost_fn is None`; calls `cost_fn(result)` otherwise. |
| `cost_fn` raises or returns a non-`Amount` | same | Logs warning at `langchain_runcycles.model_gate`; returns `estimate` so the model result is preserved. |
| Sync commit path | `model_gate.py:189` | Uses `actual = self._resolve_actual(result)` for `CommitRequest.actual`. |
| Async commit path | `model_gate.py:283` | Same. |

Built-in extractor factories in `langchain_runcycles/extractors.py` (both keyword-only on pricing args):
- `openai_cost(prompt_per_million_usd, completion_per_million_usd)`
- `anthropic_cost(input_per_million_usd, output_per_million_usd)`

Both read `AIMessage.usage_metadata` (LangChain's normalized usage shape) from `result.result[0]` and convert to `Unit.USD_MICROCENTS`. Missing/non-dict `usage_metadata`, missing token fields, negative token counts, empty `result.result`, and unrecognized shapes all raise `ValueError`, which the gate's exception-fallback path catches and converts to a commit at `estimate`.

Locked down by `tests/test_model_gate.py::test_cost_fn_used_for_commit_actual`, `::test_cost_fn_none_commits_at_estimate`, `::test_cost_fn_exception_falls_back_to_estimate`, `::test_cost_fn_not_called_in_decide_mode`, plus async siblings and the full `tests/test_extractors.py` (6 tests covering computation, edge cases, fallback paths, fractional-cent rounding, and the keyword-only-pricing guard).

## Known limitations

- **`CyclesToolGate` reserve mode commits at estimate**, not at actual usage. A `cost_fn` analogous to `CyclesModelGate.cost_fn` is the natural extension — not yet shipped. Locked down by `tests/test_tool_gate.py::test_commit_called_with_configured_estimate`.
- **Streaming verification shipped in v0.2.1.** See "Streaming contract" section below — the audit + tests in `tests/test_model_gate_streaming.py` close this item.
- **Single tenant per middleware instance** unless you supply a `SubjectExtractor` callable. Per-call subject resolution is fully supported via the callable form; only the static-Subject convenience is single-tenant.
- **Synthetic `tool_call_id` when missing.** A `ToolCallRequest` with no `id` field has its denial `ToolMessage` correlated via a fabricated `missing-<12-hex>` id, with a warning logged at `langchain_runcycles._internal`. Because the synthesis is fresh per call, the resulting idempotency key on this fallback path is *not* retry-stable. Conformant LangChain runtimes always supply `id`. Locked down by `tests/test_tool_gate.py::test_synthetic_tool_call_id_when_missing`.
- **Fan-out gate rejects per-tool action mappings.** `CyclesFanOutGate` gates *model turns*, not tool calls; a per-tool-name `Mapping` for `action` is meaningless there and is rejected at construction with `TypeError`. Locked down by `tests/test_fanout.py::test_fanout_rejects_mapping_action`.

## Streaming contract (v0.2.1+)

`CyclesModelGate` is streaming-compatible without code changes. The aggregation happens *below* the middleware layer:

| Layer | What it does |
|---|---|
| `agent.astream(...)` / `agent.ainvoke(...)` | Caller invocation. Both go through the same model-node code path. |
| `langchain/agents/factory.py:_execute_model_async` (line 1323) | The handler passed into `awrap_model_call`. Calls `await model_.ainvoke(messages)` once. |
| `BaseChatModel.ainvoke` | Internally calls `agenerate_prompt → agenerate → generate_from_stream` which consumes the model's `_astream` generator and merges all chunks into one final `AIMessage` (with summed `usage_metadata`). |
| `awrap_model_call(request, handler)` (us) | Receives the finalized `ModelResponse(result=[final_aimessage], ...)`. Calls `cost_fn(result)` exactly once. |

Implications for `CyclesModelGate`:

- `cost_fn` fires **once per model turn**, not per streamed chunk. Locked down by `tests/test_model_gate_streaming.py::test_cost_fn_called_once_when_handler_aggregates_streamed_chunks`.
- `cost_fn` reads from the **final aggregated** `AIMessage.usage_metadata`. Provider chat-model classes (`langchain-openai`, `langchain-anthropic`) accumulate per-chunk token counts inside `_astream` and stamp totals onto the final message; our extractors see those totals, not partial chunk counts. Locked down by `::test_cost_fn_sees_aggregated_usage_metadata_not_first_chunk`.
- `commit_reservation` fires **once per turn**, with `actual = cost_fn(final_result)` (or `estimate` on extractor failure).
- **Stream interruption** (consumer disconnect, `asyncio.CancelledError`) raises out of `await handler(request)` and is caught by our `except BaseException:` guard, triggering `release_reservation`. Locked down by `::test_cancellation_during_handler_releases_reservation`. CancelledError is a `BaseException` (not `Exception`) — narrowing the except clause would silently leak reservations on every cancelled stream.

Reference: `langchain==1.2.18`. If a future LangChain release passes per-chunk callbacks into `awrap_model_call` or changes the aggregation point, the regression tests will fail and we adapt.

## Settlement error policy (v0.1.2+ tool gate, v0.1.5+ model gate)

The `commit_reservation` call happens *after* the gated handler (tool call or model call) already ran, so a commit failure has two reasonable resolutions. `CyclesToolGate` and `CyclesModelGate` expose them as `settlement_error_policy`:

| Policy | Behavior |
|---|---|
| `"raise"` (default) | Surface the commit exception. Handler result is lost; caller reconciles. Strict-governance default — no cost goes unaccounted. |
| `"log"` | Log a warning, return the handler result. Reservation expires via TTL. Best-effort accounting; preferred when UX continuity matters more than per-call settlement guarantees. |

The release path (on handler-side exception) always logs and continues so the original handler exception wins; settlement_error_policy applies only to the success-path commit.

Locked down by `tests/test_tool_gate.py::test_settlement_raise_default_propagates_commit_failure`, `::test_settlement_log_swallows_commit_failure`, the corresponding `tests/test_model_gate.py` parity tests, and async siblings.

## Idempotency-key determinism (v0.1.2+)

Idempotency keys take the shape `{prefix}-{tool_call_id}` with no random component when the upstream supplies a tool call id. This is a behavior change from v0.1.0/v0.1.1, which appended a random 8-hex suffix.

The deterministic shape is the correctness story: a duplicate dispatch (durable workflow replay, middleware retry, process recovery) lands on the same Cycles reservation rather than creating a second one. UUID fallback is used only when `coerce_tool_call_id` had to synthesize a missing id — that path is non-deterministic and a known limitation (see above).

## Update protocol

Update this document when:
- A LangChain middleware API change forces a breaking signature change in this package
- A new hook is added (e.g. `wrap_model_call`, `after_model`)
- An SDK method consumed here gains/loses fields that affect the middleware contract
- Async behavior diverges between LangChain runtime versions
