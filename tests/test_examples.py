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

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("name", ["tenant_budget_agent", "multi_agent_fanout"])
def test_example_module_imports(name: str) -> None:
    path = EXAMPLES_DIR / f"{name}.py"
    assert path.exists(), f"example file missing: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main"), f"{name} missing main()"
