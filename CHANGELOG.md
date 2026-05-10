# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/runcycles/langchain-runcycles/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/runcycles/langchain-runcycles/releases/tag/v0.1.0
