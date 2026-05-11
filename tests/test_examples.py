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

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"


def _load_example(name: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{name}.py"
    assert path.exists(), f"example file missing: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["tenant_budget_agent", "multi_agent_fanout"])
def test_example_module_imports(name: str) -> None:
    module = _load_example(name)
    assert hasattr(module, "main"), f"{name} missing main()"


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
