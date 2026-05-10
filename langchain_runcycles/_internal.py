"""Internal helpers shared across middleware classes."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from typing import Any

from runcycles import Action, Amount, CyclesResponse, DecisionResponse, Subject, Unit

from langchain_runcycles._config import ActionConfig, DenialFormatter, SubjectConfig

logger = logging.getLogger(__name__)

# Conservative default — $0.0001. Tool gates in `decide` mode use this only as a
# preflight estimate (no money moves). Tool gates in `reserve` / `decide+reserve`
# modes both *reserve* and *commit* this exact amount; supply your own when you
# care about per-tool cost accuracy.
_DEFAULT_ESTIMATE = Amount(unit=Unit.USD_MICROCENTS, amount=10_000)


def coerce_tool_call_id(raw: str | None) -> str:
    """Return a non-empty tool_call_id. Fabricates one when the request lacks it.

    LangChain correlates tool calls and tool messages by id; an empty id silently
    breaks correlation. We synthesize a clearly-labeled id and log a warning so
    a curious user can grep their logs for upstream malformed requests.
    """
    if raw:
        return raw
    fabricated = f"missing-{uuid.uuid4().hex[:12]}"
    logger.warning(
        "ToolCallRequest had no 'id'; synthesized tool_call_id=%s for ToolMessage correlation.",
        fabricated,
    )
    return fabricated


def resolve_subject(config: SubjectConfig, request: Any, state: Any) -> Subject:
    """Resolve a subject from a static value or extractor callable."""
    if isinstance(config, Subject):
        return config
    return config(request, state)


def resolve_action(config: ActionConfig, request: Any, tool_name: str | None) -> Action:
    """Resolve an action from a static value, name-mapping, or extractor callable."""
    if isinstance(config, Action):
        return config
    if isinstance(config, Mapping):
        if tool_name is None:
            raise ValueError("Action mapping requires a tool name on the request.")
        try:
            return config[tool_name]
        except KeyError as e:
            raise KeyError(f"No action mapping for tool {tool_name!r}") from e
    return config(request)


def parse_decision(response: CyclesResponse) -> DecisionResponse | None:
    """Parse a /v1/decide response into a typed DecisionResponse, or None if malformed."""
    if not response.is_success or not response.body:
        return None
    try:
        return DecisionResponse.model_validate(response.body)
    except Exception:
        return None


def is_allowed(response: CyclesResponse) -> bool:
    """Return True iff the decide response indicates ALLOW or ALLOW_WITH_CAPS."""
    decision = parse_decision(response)
    return decision is not None and decision.is_allowed()


def denial_reason(response: CyclesResponse) -> str:
    """Extract a human-readable denial reason from a CyclesResponse."""
    decision = parse_decision(response)
    if decision is not None and decision.reason_code:
        return str(decision.reason_code)
    err = response.get_error_response()
    if err is not None and err.message:
        return str(err.message)
    if response.error_message:
        return str(response.error_message)
    return "denied"


def format_denial(formatter: DenialFormatter, response: CyclesResponse, tool_name: str | None) -> str:
    """Apply a DenialFormatter to a denied response."""
    if callable(formatter):
        return formatter(response)
    decision = parse_decision(response)
    return formatter.format(
        reason=denial_reason(response),
        tool=tool_name or "<unknown>",
        decision=decision.decision.value if decision is not None else "DENY",
    )


def make_idempotency_key(prefix: str, suffix: str | None = None) -> str:
    """Build an idempotency key. Deterministic when ``suffix`` is provided.

    When ``suffix`` (typically a LangChain ``tool_call_id``) is supplied, the
    returned key is ``{prefix}-{suffix}`` — same call gets the same key, so
    retries land on the same Cycles reservation rather than creating
    duplicates. This is the failure mode Cycles is built to prevent.

    A fresh UUID is used only as a last-resort fallback when no stable
    upstream identifier is available (e.g. the upstream omitted the tool
    call id and ``coerce_tool_call_id`` had to synthesize one — that
    synthesized id is itself fresh per call).
    """
    if suffix:
        return f"{prefix}-{suffix}"
    return f"{prefix}-{uuid.uuid4().hex}"


def get_tool_call(request: Any) -> dict[str, Any]:
    """Pull the tool_call dict off a ToolCallRequest (LangChain). Tolerant to dict / attr access."""
    tc = getattr(request, "tool_call", None)
    if tc is None and isinstance(request, Mapping):
        tc = request.get("tool_call")
    if not isinstance(tc, Mapping):
        raise AttributeError("ToolCallRequest is missing a 'tool_call' attribute.")
    return dict(tc)


def get_state(request: Any) -> Any:
    """Pull the state off a ToolCallRequest if present; return None if not."""
    state = getattr(request, "state", None)
    if state is None and isinstance(request, Mapping):
        state = request.get("state")
    return state


__all__ = [
    "_DEFAULT_ESTIMATE",
    "coerce_tool_call_id",
    "denial_reason",
    "format_denial",
    "get_state",
    "get_tool_call",
    "is_allowed",
    "make_idempotency_key",
    "parse_decision",
    "resolve_action",
    "resolve_subject",
]
