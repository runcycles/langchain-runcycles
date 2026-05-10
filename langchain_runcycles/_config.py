"""Public type aliases for middleware configuration.

These are the shapes users pass when constructing CyclesToolGate / CyclesFanOutGate.
Subject and action can be supplied as static values, mappings, or callables that
resolve dynamically from the tool-call request and agent state.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from runcycles import Action, Subject

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

IdempotencyNamespaceResolver: TypeAlias = Callable[[Any], str]
IdempotencyNamespace: TypeAlias = str | IdempotencyNamespaceResolver
"""Optional run/workflow/tenant scope woven into Cycles idempotency keys.
Static string or a callable receiving the request (tool gate) or state
(fan-out gate). When supplied, keys take the shape
``{prefix}-{namespace}-{tool_call_id}``; without it, the v0.1.2 shape
``{prefix}-{tool_call_id}`` is preserved (back-compat)."""
