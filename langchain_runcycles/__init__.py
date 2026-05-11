"""langchain-runcycles — Cycles middleware for LangChain agents.

Exposes three ``AgentMiddleware`` subclasses, completing the LangChain
agent governance triad of model + tool + fan-out:

* :class:`CyclesModelGate` (v0.1.5+) — gates each LLM call via the Cycles
  SDK (``decide``, ``reserve``, or both). Returns a ``ModelResponse`` whose
  ``AIMessage`` carries the denial reason on deny, so the agent terminates
  naturally.
* :class:`CyclesToolGate` — gates each tool call. Returns a ``ToolMessage``
  on denial so the model can recover.
* :class:`CyclesFanOutGate` — caps model turns per run and optionally
  consults ``decide()`` on each turn so an external policy service can halt
  fan-out.

All three work with sync or async Cycles clients; pair the right client
with ``.invoke()`` / ``.ainvoke()`` on the LangChain agent. Compose them in
a single ``middleware=[...]`` list — natural ordering is fan-out → model
→ tool so runaway loops halt before model spend, and model spend reserves
before tool side effects.
"""

from langchain_runcycles._config import (
    ActionConfig,
    ActionExtractor,
    ActionMap,
    CostFn,
    DenialFormatter,
    IdempotencyNamespace,
    IdempotencyNamespaceResolver,
    SubjectConfig,
    SubjectExtractor,
    TurnCounter,
)
from langchain_runcycles.fanout import CyclesFanOutGate
from langchain_runcycles.model_gate import CyclesModelGate
from langchain_runcycles.tool_gate import CyclesToolGate, Mode, SettlementErrorPolicy

__all__ = [
    "ActionConfig",
    "ActionExtractor",
    "ActionMap",
    "CostFn",
    "CyclesFanOutGate",
    "CyclesModelGate",
    "CyclesToolGate",
    "DenialFormatter",
    "IdempotencyNamespace",
    "IdempotencyNamespaceResolver",
    "Mode",
    "SettlementErrorPolicy",
    "SubjectConfig",
    "SubjectExtractor",
    "TurnCounter",
]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

try:
    __version__ = _metadata_version("langchain-runcycles")
except PackageNotFoundError:  # pragma: no cover - editable install before metadata is built
    __version__ = "0.0.0+local"
