# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.4]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/runcycles/langchain-runcycles/releases/tag/v0.1.0
