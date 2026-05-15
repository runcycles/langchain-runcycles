"""Public type aliases for middleware configuration.

These are the shapes users pass when constructing CyclesModelGate /
CyclesToolGate / CyclesFanOutGate. Subject and action can be supplied as
static values, mappings (action only — subject is static or callable, not
a mapping), or callables that resolve dynamically from the request and
agent state. ``CostFn`` is the per-call extractor type for
``CyclesModelGate.cost_fn`` (v0.2.0+). ``ToolCostFn`` is the two-argument
extractor type for ``CyclesToolGate.cost_fn`` (v0.3.0+).
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from runcycles import Action, Amount, Subject

SubjectExtractor: TypeAlias = Callable[[Any, Any], Subject]
"""(request, state) -> Subject. The request is the LangChain ToolCallRequest in tool gates,
or the AgentState in fan-out gates; users get whichever is most useful for their case."""

SubjectConfig: TypeAlias = Subject | SubjectExtractor

ActionExtractor: TypeAlias = Callable[[Any], Action]
ActionMap: TypeAlias = Mapping[str, Action]
ActionConfig: TypeAlias = Action | ActionMap | ActionExtractor
"""Either a single static Action, a name-to-Action mapping (dict keyed by tool name),
or a callable that derives the Action from the request."""

DenialFormatter: TypeAlias = str | Callable[[Any], str]
"""Either a format string with {reason}/{tool}/{decision} placeholders,
or a callable taking the CyclesResponse and returning a denial message."""

TurnCounter: TypeAlias = Callable[[Any], int]
"""(state) -> turn count. Default counts AIMessages; override for custom semantics."""

IdempotencyNamespaceResolver: TypeAlias = Callable[[Any], str | None]
IdempotencyNamespace: TypeAlias = str | IdempotencyNamespaceResolver
"""Optional run/workflow/tenant scope woven into Cycles idempotency keys.

Static string or a callable receiving the request (tool gate) or state
(fan-out gate). The callable may return ``None`` to disable namespacing
for that particular call (per-call opt-out — useful when some calls
should be globally scoped while others are run-scoped).

When a non-empty namespace is in effect, keys take the shape
``{prefix}-{namespace}-{tool_call_id}``; otherwise the v0.1.2 shape
``{prefix}-{tool_call_id}`` is preserved (back-compat)."""

CostFn: TypeAlias = Callable[[Any], Amount]
"""(model_response) -> Amount. Per-call cost extractor for ``CyclesModelGate`` (v0.2.0+).

When supplied, ``CyclesModelGate`` calls ``cost_fn(result)`` after the model
handler returns and uses the returned ``Amount`` for ``commit_reservation``
instead of the configured ``estimate``. The callable receives the
``ModelResponse`` returned by the wrapped handler so the extractor can pull
token counts from the response's ``AIMessage.usage_metadata`` (or
``response_metadata``) and convert to the unit your Cycles budgets use.

If the callable raises or returns a non-``Amount``, the gate logs a warning
and falls back to the configured ``estimate`` so a costing bug never erases a
successful model result. Use the provided ``langchain_runcycles.extractors``
factories for OpenAI / Anthropic shapes, or write your own."""

ToolCostFn: TypeAlias = Callable[[Any, Any], Amount]
"""(tool_call_request, tool_result) -> Amount.

Per-call cost extractor for ``CyclesToolGate`` (v0.3.0+). The alias uses
``Any`` for both arguments so the public config surface does not depend on
LangGraph's runtime request class. At runtime, the first argument is the
LangChain ``ToolCallRequest`` so extractors can inspect the tool name,
arguments, call id, and state. The second argument is the wrapped handler's
result, usually a ``ToolMessage`` but possibly any value returned by the
runtime or tests.

When supplied, ``CyclesToolGate`` calls ``cost_fn(request, result)`` after the
tool handler returns and uses the returned ``Amount`` for
``commit_reservation`` instead of the configured ``estimate``. If the callable
raises or returns a non-``Amount``, the gate logs a warning and falls back to
the configured ``estimate`` so a costing bug never erases a successful tool
result."""
