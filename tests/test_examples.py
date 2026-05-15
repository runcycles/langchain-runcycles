"""Smoke tests for the bundled example scripts.

Imports each example as a module (without running `main()`) and verifies it
loads cleanly. This catches:
  - import-level errors when LangChain's API moves (e.g. a renamed middleware)
  - syntax mistakes that slipped past `ruff check`
  - module-level side effects that fail without external state

LangChain's docs-PR criteria explicitly require example code to run without
errors; these tests provide a CI guard so the examples don't drift behind the
implementation.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"


def _load_example(name: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{name}.py"
    assert path.exists(), f"example file missing: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["tenant_budget_agent", "tool_cost_fn", "multi_agent_fanout"])
def test_example_module_imports(name: str) -> None:
    module = _load_example(name)
    assert hasattr(module, "main"), f"{name} missing main()"


def test_tool_cost_fn_reads_json_serialized_tool_message_content() -> None:
    module = _load_example("tool_cost_fn")
    request = _ToolCostRequest("lookup_customer")
    result = ToolMessage(
        content='{"email": "alice@example.com", "charged_microcents": 12345}',
        tool_call_id="tc_lookup",
    )

    amount = module.tool_cost(request, result)
    assert amount.amount == 12_345


def test_tool_cost_fn_falls_back_when_result_content_is_not_structured() -> None:
    module = _load_example("tool_cost_fn")
    request = _ToolCostRequest("lookup_customer")
    result = ToolMessage(content="not json", tool_call_id="tc_lookup")

    amount = module.tool_cost(request, result)
    assert amount.amount == 12_500


def test_tool_cost_fn_prices_sms_segments_from_request_args() -> None:
    module = _load_example("tool_cost_fn")
    request = _ToolCostRequest("send_sms", {"body": "x" * 161})

    amount = module.tool_cost(request, "Sent")
    assert amount.amount == 150_000


def test_multi_agent_demo_preserves_tenant_state_and_gate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example("multi_agent_fanout")
    captured: dict[str, Any] = {}

    def fake_create_agent(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "agent"

    monkeypatch.setenv("CYCLES_API_KEY", "test-key")
    monkeypatch.setattr(module, "create_agent", fake_create_agent)

    model = object()
    assert module.build_agent(model=model) == "agent"
    assert captured["model"] is model
    assert captured["state_schema"] is module.ResearchAgentState

    subject = module.per_tenant_subject(None, {"config": {"tenant": "globex"}})
    assert subject.tenant == "globex"

    middleware_names = [type(m).__name__ for m in captured["middleware"]]
    assert middleware_names == [
        "CyclesFanOutGate",
        "CyclesModelGate",
        "HumanInTheLoopMiddleware",
        "CyclesToolGate",
    ]
    model_gate = captured["middleware"][1]
    assert model_gate._mode == "decide+reserve"  # noqa: SLF001


class _ToolCostRequest:
    def __init__(self, name: str, args: dict[str, Any] | None = None) -> None:
        self.tool_call = {"name": name, "args": args or {}, "id": "tc_1"}
