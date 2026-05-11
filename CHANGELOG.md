# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Docs

- **README "Settlement (commit) failures" and `docs/runcycles.mdx` "Settlement-failure policy"** now mention `CyclesModelGate` parity and the non-success-`CyclesResponse` failure path. The prose described `settlement_error_policy` as tool-only and exception-only; v0.2.3 already applies it to both gates and both failure modes (raised exceptions and HTTP-failure responses). `AUDIT.md` and `CHANGELOG.md` were already accurate; this is a prose-drift correction with no behavior change.

## [0.2.3] - 2026-05-11

Correctness patch driven by external review. One real governance bug, two doc fixes, one minor lockdown. No public API change.

### Fixed

- **`commit_reservation` and `release_reservation` HTTP-failure responses are now honored** (both gates, sync + async). The runcycles SDK returns `CyclesResponse.http_error(...)` on HTTP failures *without raising*; v0.2.0–v0.2.2 only caught raised exceptions, so a failed commit silently looked like a successful commit and bypassed the documented `settlement_error_policy` contract. The release-path was similarly silent on HTTP-failure responses. Both paths now check `response.is_success` and apply the same policy as the exception path:
  - Commit HTTP failure + `settlement_error_policy="raise"` (default) → `RuntimeError` carrying `denial_reason(response)`.
  - Commit HTTP failure + `settlement_error_policy="log"` → warning logged, handler result preserved.
  - Release HTTP failure (best-effort) → warning logged, never raised (regardless of policy).
  
  Log message wording changed from "commit failed" to "commit raised" / "commit returned HTTP failure" so operators can distinguish the two failure modes in audit logs. Locked down by `tests/test_tool_gate.py::test_commit_http_failure_raise_default_propagates`, `::test_commit_http_failure_log_swallows`, `::test_release_http_failure_logged`, sync/async equivalents in `test_model_gate.py` / `test_model_gate_async.py` / `test_tool_gate_async.py`. **This is the user-visible behavior change in this release** — code that relied on the silent-success path will now see governance-policy responses (which is what was always documented).

- **`CyclesToolGate` with `tool_call={"id": None}`** now flows through `coerce_tool_call_id`'s synthesis path and produces `missing-<12-hex>`. Previously the `str(tool_call.get("id", "")) or None` pattern produced the literal string `"None"`, creating deterministic `decide-None` idempotency-key collisions across all malformed calls. Locked down by `tests/test_tool_gate.py::test_explicit_none_tool_call_id_uses_synthetic_path` and the async sibling.

### Docs

- **`docs/runcycles.mdx` `CyclesToolGate` section** — clarified that `action` is a single-argument callable `(request) -> Action` while `subject` is a two-argument callable `(request, state) -> Subject`. The prior wording said both "resolve from request and state," which would TypeError if users followed it literally for `action`.
- **`examples/tenant_budget_agent.py` install instructions** — corrected `langchain-openai` / `OPENAI_API_KEY` to `langchain-anthropic` / `ANTHROPIC_API_KEY` to match the `claude-sonnet-4-6` model the example actually uses.

### Coverage

159 tests (was 145), 99.62% coverage (gate `fail_under = 95`); `model_gate.py` and `tool_gate.py` both at 100%. 14 new tests cover the four corrected paths (3 HTTP-failure scenarios × 4 gate variants + 2 None-id regression checks).

### Behavior change

Existing code that relied on silent-success commit on HTTP-failure responses will now invoke `settlement_error_policy`. The default policy is `"raise"`, so callers that never set the policy explicitly will start seeing `RuntimeError("Cycles commit_reservation returned HTTP failure...")` propagate when the Cycles server returns a non-2xx commit response. This is the documented contract; the prior silent behavior was a bug. Set `settlement_error_policy="log"` to opt into the swallow-and-preserve-result behavior if that matches your UX needs.

## [0.2.2] - 2026-05-11

Final piece of [issue #13](https://github.com/runcycles/langchain-runcycles/issues/13): the multi-agent / HITL / multi-tenant demo that meets LangChain's co-marketing bar. No library code change; example + write-up only.

### Added

- **`examples/multi_agent_fanout.py`** rewritten to compose the **full v0.2.x governance triad**: `CyclesFanOutGate` + `CyclesModelGate` (with `anthropic_cost` extractor for actual-cost commits) + LangChain's `HumanInTheLoopMiddleware` on `send_email` + `CyclesToolGate` (in `decide+reserve` mode). Composition order is documented inline to match LangChain's hook timing (fan-out → model → HITL review → final tool authorization). A custom state schema preserves per-tenant config so the same agent process serves many tenants with independent budgets.
- **`examples/multi_agent_fanout_writeup.md`** — a problem-framed pattern walkthrough leading with "pre-execution budget authority for multi-tenant agents," with the demo as a concrete proof point. Three failure-mode scenarios (full-budget tenant, fully-denied tenant, partially-allowed tenant) demonstrate how each gate independently saves the cost of everything downstream. Companion artifact for LangChain co-marketing per their stated "educational, problem-framed, multi-agent / HITL" preference.

### Changed

- **README "Examples" section** now describes the demo with all three gates named, links the write-up.

### Coverage

145 tests, 99.59% coverage (gate `fail_under = 95`). `tests/test_examples.py` import-smoke continues to gate both bundled examples on every CI run and now verifies the demo's tenant state schema, middleware order, and `decide+reserve` model gate mode.

### Closes

Issue #13 in full: parts 1 (cost_fn + extractors), 2 (streaming verification), and 3 (demo agent + write-up) all landed across v0.2.0, v0.2.1, and v0.2.2. v0.2.x line is feature-complete for the original v0.2.0 scope.

## [0.2.1] - 2026-05-11

Streaming-path verification for `CyclesModelGate` — closes part 2 of [issue #13](https://github.com/runcycles/langchain-runcycles/issues/13). No code changes; tests + docs only.

### Verified

`CyclesModelGate` is streaming-compatible without code changes because LangChain's streaming aggregation happens *inside* `BaseChatModel.ainvoke`, which is called by `_execute_model_async` (`langchain/agents/factory.py:1323`) — the handler our `awrap_model_call` receives. The handler aggregates all streamed chunks into one final `AIMessage` with summed `usage_metadata` before returning the `ModelResponse`. Middleware never sees per-chunk callbacks.

### Added

- **`tests/test_model_gate_streaming.py`** (3 tests) — regression checks locking down the streaming contract on our side:
  - `test_cost_fn_called_once_when_handler_aggregates_streamed_chunks` — handler internally aggregates 4 chunks into one final message; `cost_fn` fires exactly once. A regression that triggered per-chunk commits would fail this test.
  - `test_cost_fn_sees_aggregated_usage_metadata_not_first_chunk` — handler sums chunk `usage_metadata` into the final message; the `openai_cost` extractor computes the commit `actual` from the *summed* totals (250_000 microcents), not the first chunk's partial counts (which would be 50_000). Explicit `!=` assertion so a "first-chunk-only" regression fails loudly.
  - `test_cancellation_during_handler_releases_reservation` — `asyncio.CancelledError` raised inside the handler (consumer disconnects mid-stream) triggers `release_reservation`. CancelledError is a `BaseException`, not `Exception` — this test guards against a future refactor that narrows the `except` clause and silently leaks reservations.

### Coverage

144 tests, 99.59% coverage (gate `fail_under = 95`); `model_gate.py`, `extractors.py` both at 100%.

### Behavior change

None. Pure verification + audit. Existing v0.2.0 callers see no change.

### AUDIT.md

New "Streaming contract (v0.2.1+)" section documents the layered call path from `agent.astream(...)` down to our `awrap_model_call`, with the load-bearing fact: aggregation lives *below* the middleware layer, so we only ever see finalized responses.

### Deferred to a later v0.2.x

- **`examples/multi_agent_fanout.py`** — multi-tenant fan-out + HITL demo agent meeting LangChain's stated co-marketing bar. Tracked separately on #13.

## [0.2.0] - 2026-05-11

Per-call actual-cost extraction for `CyclesModelGate` — closes the v0.1.x "commits at estimate" gap that prompted the [v0.2.0 issue](https://github.com/runcycles/langchain-runcycles/issues/13). Reserve-mode and `decide+reserve`-mode model calls can now debit Cycles budgets at the model's actual reported token usage, not the worst-case estimate.

### Added

- **`cost_fn` parameter on `CyclesModelGate`.** Optional `Callable[[ModelResponse], Amount]`. When supplied, the middleware calls `cost_fn(result)` after the wrapped model handler returns and uses the returned `Amount` for `commit_reservation` instead of the configured `estimate`. When unset, behavior is identical to v0.1.x (commit-at-estimate). The callable receives the LangChain `ModelResponse` returned by the handler so extractors can pull `usage_metadata` off the contained `AIMessage`.
- **`langchain_runcycles.extractors` module** with two factory functions:
  - `openai_cost(prompt_per_million_usd=..., completion_per_million_usd=...)` — returns a `cost_fn` that reads `AIMessage.usage_metadata` and converts to `USD_MICROCENTS` using OpenAI-labeled pricing (prompt/completion).
  - `anthropic_cost(input_per_million_usd=..., output_per_million_usd=...)` — same shape, Anthropic-labeled pricing (input/output). Both factories take keyword-only pricing args so callers can't accidentally swap input and output rates.
- **`CostFn` type alias** exported from the package root for type-annotating user-supplied extractors.

### Resilience

- **`cost_fn` errors never erase the model result.** If `cost_fn(result)` raises or returns a non-`Amount`, `CyclesModelGate` logs a warning and falls back to the configured `estimate` for the commit. The model result is still returned to the agent. This means a stale or wrong extractor downgrades the debit accuracy (estimate vs. actual) but never breaks the agent loop. Locked down by `tests/test_model_gate.py::test_cost_fn_exception_falls_back_to_estimate`, `::test_cost_fn_invalid_return_falls_back_to_estimate`, and the async siblings.

### Coverage

141 tests, 99.59% coverage (gate `fail_under = 95`); `model_gate.py` and `extractors.py` both at 100%. Cost-fn coverage now includes applied / None-fallback / exception-fallback / invalid-return fallback / decide+reserve parity / not-called-in-decide-mode across sync and async paths, plus missing-reservation-id async coverage and awaitable-handler reserve-mode coverage on `CyclesModelGate`. Extractor tests cover OpenAI/Anthropic shape extraction, zero-token edge cases, missing-usage-metadata fallback, missing token fields, negative token counts, empty-result fallback, fractional-cent rounding, and the keyword-only pricing-arg guard.

### Behavior change

`cost_fn` is purely additive. Callers on v0.1.x who don't pass it see no change. Callers who pass it get actual-cost commits instead of estimate-cost commits — a more accurate debit, never a denial path change.

### Deferred to a later v0.2.x

- **Streaming integration verification.** `wrap_model_call` runs once per turn even when the underlying call streams; we need explicit tests confirming the commit fires after the stream is fully consumed. Tracked separately on #13.
- **`examples/multi_agent_fanout.py`** — the multi-tenant fan-out + HITL demo agent that meets LangChain's stated co-marketing bar. Tracked separately on #13.

## [0.1.6] - 2026-05-10

Doc cleanup pass. v0.1.5 shipped `CyclesModelGate` but several places in the repo still described the package as "two middleware classes" or "no model-call middleware yet." External review caught the staleness; this release brings the user-visible metadata in line with what the v0.1.5 code actually does. No code changes; behavior is identical to v0.1.5.

### Fixed

- **`pyproject.toml` description** now reads "pre-execution budget authority for model calls, tool calls, and runaway agent loops" (was "pre-tool-call authorization, fan-out caps, and per-tenant budget enforcement"). PyPI listing now correctly advertises the v0.1.5 surface.
- **`langchain_runcycles/__init__.py` module docstring** now describes three `AgentMiddleware` subclasses (was two), with composition guidance (fan-out → model → tool ordering).
- **README "Known limitations"**: replaced "No model-call middleware yet" with the accurate "architecture-complete but commits at estimate" framing — `CyclesModelGate` exists, just doesn't yet extract actual provider token usage.
- **`docs/runcycles.mdx`**: opening paragraph updated from "two classes" to "three classes"; Details metadata table now lists `CyclesModelGate` alongside `CyclesToolGate` and `CyclesFanOutGate`; "v0.1.x scope" subsection rewritten to honestly distinguish architecture milestone (v0.1.5) from production polish (v0.2.0).
- **`AUDIT.md`** Audit Scope section now lists model hooks (`wrap_model_call` / `awrap_model_call`); SDK methods table now references `model_gate.py` alongside `tool_gate.py` for `decide` / `create_reservation` / `commit_reservation` / `release_reservation`.

### Behavior change

None. Doc cleanup only. v0.1.5 callers' code unchanged.

## [0.1.5] - 2026-05-10

Adds `CyclesModelGate` — pre-model-call authorization middleware — closing the third leg of the LangChain agent governance triad. Closes #10.

`CyclesToolGate` already gated tool calls; `CyclesFanOutGate` already capped model turns; this release adds the third middleware so model calls themselves are intercepted via LangChain's `wrap_model_call` hook. With v0.1.5, the package can truthfully say it puts pre-execution authority in front of model calls, tool calls, and runaway agent loops.

This is the **architecture milestone** — feature parity with `CyclesToolGate`. The **production milestone** (v0.2.0) will add provider-specific token-cost extractors, streaming integration, and a polished demo agent.

### Added

- **`CyclesModelGate`** — new `AgentMiddleware` subclass overriding `wrap_model_call` (sync) and `awrap_model_call` (async). On denial in `decide` mode, returns a `ModelResponse` whose `AIMessage` carries the denial reason (the agent terminates naturally because the AIMessage has no `tool_calls`).
- **All three modes** (`decide` / `reserve` / `decide+reserve`) at parity with `CyclesToolGate`.
- **`settlement_error_policy`** parity (default `"raise"`).
- **`idempotency_namespace`** parity (static or callable, callable receives the `ModelRequest`).
- **Public API exports**: `CyclesModelGate` re-exported from `langchain_runcycles`.

### Changed

- README "What's in the box" gains a third bullet for `CyclesModelGate`.
- AUDIT.md hooks table gains `wrap_model_call` / `awrap_model_call` rows; "No model-call middleware yet" line removed from Known Limitations.
- `docs/runcycles.mdx` gets a third middleware section + a 3-class composition example.

### Known limitations (carried into v0.2.0)

- **Commits at the configured `estimate`**, not actual token cost. Provider-specific token extraction (OpenAI, Anthropic) is v0.2.0 scope. For precise per-call actual-cost capture today, use the callback handler from `cycles-client-python` instead, or until v0.2.0 ships.
- **No streaming integration**. For streaming LLM calls, use `runcycles.stream_reservation` directly. v0.2.0 may add streaming support inside `wrap_model_call`.
- **Per-call key uses UUID fallback**. Model-call requests don't carry an upstream-stable id like `tool_call_id`, so each call gets a fresh UUID for the per-call slot. Namespacing (`model-decide-{namespace}-{32-hex}`) still scopes by run/workflow/tenant. v0.2.0 may extract a turn-id or message-hash for full retry-stability.

### Test coverage

- 28 new tests (115 total, was 87). Coverage 99.07% (gate 95%).
- Sync + async parity for all paths; dedicated tests for the deny-path `ModelResponse` shape.

### Backward compatibility

Purely additive. v0.1.4 callers' code unchanged.

## [0.1.4] - 2026-05-10

External review of v0.1.3 caught two real defects (stale README dep line, type alias narrower than the documented per-call opt-out) plus a docs-example issue (undefined helper in a runnable-looking snippet). All fixed; no functional change.

### Fixed

- **Stale `langchain-core >= 0.3` line in README's Requirements section.** `pyproject.toml` was correctly tightened to `>=1.0,<2.0` in v0.1.2 but the README wasn't updated at the time. Now consistent.
- **`IdempotencyNamespaceResolver` type alias widened from `Callable[[Any], str]` to `Callable[[Any], str | None]`.** Runtime already supported a callable returning `None` (per-call opt-out, documented since v0.1.3 and tested by `test_callable_namespace_returning_none_opts_out_per_call`). The type signature was narrower than the documented contract; strict-mypy users would have hit a false negative when writing the documented opt-out callable. No behavior change.
- **Docs idempotency-namespace example now uses `request.state["run_id"]`** instead of an undefined `current_run_id()` / `current_run_id_contextvar.get(...)` helper. The new form is runnable against a real LangChain `ToolCallRequest` and matches the LangChain idiom for accessing per-call state. README, `docs/runcycles.mdx`, and the langchain-ai/docs PR mirror all updated together.

### Behavior change

None. Type widening is permissive (the old narrower type was a strict subset of the new one), README and docs are user-facing prose. v0.1.3 callers' code keeps working unchanged.

## [0.1.3] - 2026-05-10

Adds run / workflow / tenant scoping to Cycles idempotency keys, addressing the v0.1.2 review concern about cross-run collision when frameworks reuse short tool call ids like `tc_1`. Backward-compatible — keys without a configured namespace keep the v0.1.2 shape exactly. Closes #6.

### Added

- **`idempotency_namespace` config on `CyclesToolGate` and `CyclesFanOutGate`.** Optional `str | Callable[[Any], str]`. When supplied, every Cycles idempotency key becomes `{prefix}-{namespace}-{tool_call_id}` (or `{prefix}-{namespace}-{32-hex}` in the fanout case where there's no per-call upstream id). The callable form receives the LangChain `ToolCallRequest` for tool gates and the agent `state` for fan-out gates — useful for extracting a workflow run id, tenant id, or other run-scoped context.
- **`make_idempotency_key` now accepts a `namespace` keyword-only argument.** Same shape semantics as the middleware-level config; useful when calling the helper directly from custom integrations.
- **Public type aliases** `IdempotencyNamespace` and `IdempotencyNamespaceResolver` re-exported from the package root for users typing their config callables.

### Changed

- **AUDIT.md idempotency-key section** updated with the new four-shape table (namespace+suffix, suffix only, namespace only, neither) and cross-references to the namespace tests.
- **README** gains an "Idempotency-key namespacing" subsection under Configuration showing static and callable forms with a practical run-id example.
- **MDX Production notes** mirrors the README addition (will land via the same content sync that keeps the langchain-ai/docs PR in step).

### Backward compatibility

No behavior change for users not setting `idempotency_namespace`. Locked down by `tests/test_tool_gate.py::test_no_namespace_preserves_v012_key_shape`. New tests covering static, callable, fanout-state-derived, and cross-run-collision-prevention paths bring the suite to 85 tests, ≥99% coverage.

## [0.1.2] - 2026-05-10

Credibility patch addressing external review feedback. Three real correctness/copy issues that landed in the v0.1.0/v0.1.1 cuts; nothing is co-marketed before these are fixed.

### Changed

- **README copy: removed `wrap_model_call` overclaim.** Earlier copy listed `wrap_model_call` alongside the implemented hooks and said the package runs "before LLM calls or tool actions execute" — but `wrap_model_call` is not implemented in v0.1.x. The README now accurately scopes coverage to `wrap_tool_call` (tool-call authorization) and `before_model` (fan-out caps), with an explicit note that model-call middleware is on the roadmap.
- **Brand: title and copy use "Cycles", not "Runcycles".** "Cycles" is the product; `runcycles` is just the domain. H1 changed from `LangChain Runcycles — ...` to `Cycles for LangChain — AI agent middleware for budget and action authority`. Same fix in `docs/runcycles.mdx` (title `Cycles middleware integration`).
- **Copy: dropped the "Works with LangGraph, LangSmith, OpenAI, Anthropic, MCP servers..." framework name-drop.** Replaced with `Provider-neutral: works with any LangChain 1.x agent regardless of model provider, as long as actions flow through LangChain middleware/tool execution.` More accurate for a middleware package that's truly model-agnostic at this layer.
- **Tightened `langchain-core` dep** from `>=0.3` to `>=1.0,<2.0` per review feedback. `langchain >=1.0,<2.0` already pulls `langchain-core 1.x` transitively; the explicit pin avoids a stale 0.x landing accidentally in mixed environments.
- **Settlement docs: added a retry-duplicate-side-effect callout.** When `settlement_error_policy="raise"` propagates the commit failure, a LangChain agent may retry — at which point the tool's side effect (email, payment, write) repeats. README and the MDX Production-notes section now flag this so users can choose `"log"` deliberately for non-idempotent tool side effects.

### Fixed

- **Idempotency keys are now deterministic per `tool_call_id`.** Previous shape was `{prefix}-{tool_call_id}-{8-hex-uuid}`, which made every retry of the same tool call land on a *new* Cycles reservation — defeating the point of idempotency. New shape is `{prefix}-{tool_call_id}` so a duplicate dispatch (durable workflow replay, middleware retry, process recovery) lands on the same reservation. Random UUIDs are used only as a last-resort fallback when the upstream omits `tool_call_id`. Locked down by `tests/test_tool_gate.py::test_idempotency_keys_are_deterministic_per_tool_call_id` and `::test_idempotency_key_retry_lands_on_same_key`.

### Added

- **`settlement_error_policy` config on `CyclesToolGate`.** New `Literal["raise", "log"]` parameter controlling what happens if the post-tool-run `commit_reservation` call itself fails. New default `"raise"` propagates the commit failure so the caller can reconcile (governance-first); the previous v0.1.0/v0.1.1 behavior is opt-in via `"log"`. Documented tradeoff in README "Settlement (commit) failures" subsection. Tests: `test_settlement_raise_default_propagates_commit_failure`, `test_settlement_log_swallows_commit_failure`, plus async siblings and `test_invalid_settlement_policy_raises`.

### Behavior change (minor)

`settlement_error_policy` defaults to `"raise"`. Users on v0.1.0/v0.1.1 who relied on the swallowed-commit-failure behavior should explicitly pass `settlement_error_policy="log"`. Default chosen because for a *governance* package, silently dropping accounting on commit failure is more dangerous than surfacing the error.

## [0.1.1] - 2026-05-10

Discovery / SEO refresh. No code changes; metadata-only release that aligns the package's PyPI listing and README with the parent SDK (`cycles-client-python`) for category-search and ecosystem discovery.

### Changed

- `pyproject.toml`: rewrote `description` to append "using create_agent"; expanded `keywords` from 17 to 21 (added `agent-safety`, `llm-cost`, `llmops`, `action-control`, `mcp` matching the parent SDK's v0.4.1 SEO refresh); repointed `Documentation` URL off the placeholder onto the GitHub README anchor. (#3)
- `README.md`: aligned badges (3 → 5, added PyPI Downloads + Coverage), H1 title (now `LangChain Runcycles — AI agent middleware for budget and action authority`), and bolded keyword-dense hook with the cadence used by `cycles-client-python` and `cycles-client-typescript` so the GitHub social card, search snippets, and PyPI page lead with the same category-search keywords. (#4)
- GitHub repo metadata (description, homepage, 20 topics) was set out-of-band via `gh repo edit` to match the topic set the sibling SDKs use, plus LangChain-specific topics (`langchain`, `langgraph`, `agent-middleware`).

## [0.1.0] - 2026-05-10

Initial public release. First-class LangChain agent middleware integration for Cycles, exposing pre-tool-call authorization and fan-out enforcement on top of the `runcycles` SDK.

### Added

- `CyclesToolGate` middleware (sync + async) — gates tool calls via `wrap_tool_call`. Supports `decide`, `reserve`, and `decide+reserve` modes against the Cycles SDK. Returns a `ToolMessage` with a denial reason when authorization fails; reserves and commits/releases budget around allowed calls.
- `CyclesFanOutGate` middleware (sync + async) — enforces a per-run turn cap via `before_model` with `@hook_config(can_jump_to=["end"])`. Halts the agent with an `AIMessage` and `jump_to: "end"` when the cap is reached.
- Subject extractor and action mapper config — both static (`Subject`, dict) and dynamic (`Callable`) shapes supported.
- Examples: `tenant_budget_agent.py` (tenant cap + risky-tool denial) and `multi_agent_fanout.py` (multi-agent / HITL flow).
- `AUDIT.md` documenting LangChain middleware API conformance (hooks, ToolMessage shape, jump_to semantics, SDK methods consumed).

[0.2.3]: https://github.com/runcycles/langchain-runcycles/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/runcycles/langchain-runcycles/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/runcycles/langchain-runcycles/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/runcycles/langchain-runcycles/releases/tag/v0.1.0
