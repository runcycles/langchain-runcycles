# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-10

Initial public release. First-class LangChain agent middleware integration for Cycles, exposing pre-tool-call authorization and fan-out enforcement on top of the `runcycles` SDK.

### Added

- `CyclesToolGate` middleware (sync + async) — gates tool calls via `wrap_tool_call`. Supports `decide`, `reserve`, and `decide+reserve` modes against the Cycles SDK. Returns a `ToolMessage` with a denial reason when authorization fails; reserves and commits/releases budget around allowed calls.
- `CyclesFanOutGate` middleware (sync + async) — enforces a per-run turn cap via `before_model` with `@hook_config(can_jump_to=["end"])`. Halts the agent with an `AIMessage` and `jump_to: "end"` when the cap is reached.
- Subject extractor and action mapper config — both static (`Subject`, dict) and dynamic (`Callable`) shapes supported.
- Examples: `tenant_budget_agent.py` (tenant cap + risky-tool denial) and `multi_agent_fanout.py` (multi-agent / HITL flow).
- `AUDIT.md` documenting LangChain middleware API conformance (hooks, ToolMessage shape, jump_to semantics, SDK methods consumed).

[0.1.0]: https://github.com/runcycles/langchain-runcycles/releases/tag/v0.1.0
