"""langchain-runcycles — Cycles middleware for LangChain agents.

Exposes two ``AgentMiddleware`` subclasses:

* :class:`CyclesToolGate` — gates each tool call via the Cycles SDK
  (``decide``, ``reserve``, or both). Returns a ``ToolMessage`` on denial so
  the model can recover.
* :class:`CyclesFanOutGate` — caps model turns per run and optionally consults
  ``decide()`` on each turn so an external policy service can halt fan-out.

Both classes work with sync or async Cycles clients; pair the right client
with ``.invoke()`` / ``.ainvoke()`` on the LangChain agent.
"""

from langchain_runcycles._config import (
    ActionConfig,
    ActionExtractor,
    ActionMap,
    DenialFormatter,
    IdempotencyNamespace,
    IdempotencyNamespaceResolver,
    SubjectConfig,
    SubjectExtractor,
    TurnCounter,
)
from langchain_runcycles.fanout import CyclesFanOutGate
from langchain_runcycles.tool_gate import CyclesToolGate, Mode, SettlementErrorPolicy

__all__ = [
    "ActionConfig",
    "ActionExtractor",
    "ActionMap",
    "CyclesFanOutGate",
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
