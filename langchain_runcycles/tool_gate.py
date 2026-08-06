"""CyclesToolGate — pre-tool-call authorization middleware for LangChain agents.

Wraps every tool call with a Cycles authorization check. Three modes:

  * ``"decide"`` — call ``client.decide()`` and deny on a non-allow decision.
    No reservation, no commit. Lowest overhead; pure policy gate.
  * ``"reserve"`` — create a reservation, run the tool, commit on success
    or release on exception. Used when the tool consumes budget that should
    be debited.
  * ``"decide+reserve"`` — authorize via decide() first, then reserve+commit.
    Most strict; useful when policy and budget are evaluated separately.

Subject and action are resolved per call from constructor config (static value,
name-to-Action mapping, or callable). Denial returns a ``ToolMessage`` so the
model can recover gracefully; the underlying tool handler is never invoked.

v0.3.0+ supports per-call actual-cost extraction via optional ``cost_fn``:
if supplied, the callable receives ``(ToolCallRequest, result)`` and returns
an ``Amount`` used for the commit. If it raises or returns a non-``Amount``,
the gate logs a warning and falls back to the configured ``estimate`` so the
tool result is preserved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from runcycles import (
    Amount,
    AsyncCyclesClient,
    CyclesClient,
    CyclesProtocolError,
    DecisionRequest,
)

from langchain_runcycles._config import (
    ActionConfig,
    DenialFormatter,
    IdempotencyNamespace,
    SubjectConfig,
    ToolCostFn,
)
from langchain_runcycles._internal import (
    _DEFAULT_ESTIMATE,
    coerce_tool_call_id,
    format_denial,
    get_state,
    get_tool_call,
    is_allowed,
    make_idempotency_key,
    resolve_action,
    resolve_subject,
    response_from_protocol_error,
)


def _coerce_id(value: Any) -> str:
    """Treat None, empty, or non-string ids the same as missing — fall through to coerce_tool_call_id's synthesis path.

    Without this, an upstream `tool_call={"id": None}` would `str(None)` to the literal
    string "None" (truthy non-empty), bypassing synthesis and producing deterministic
    `decide-None` collisions across malformed calls."""
    if isinstance(value, str) and value:
        return coerce_tool_call_id(value)
    return coerce_tool_call_id(None)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_VALID_MODES: tuple[str, ...] = ("decide", "reserve", "decide+reserve")
_VALID_SETTLEMENT_POLICIES: tuple[str, ...] = ("raise", "log")
Mode = Literal["decide", "reserve", "decide+reserve"]
SettlementErrorPolicy = Literal["raise", "log"]


class CyclesToolGate(AgentMiddleware):
    """Authorize each LangChain agent tool call against Cycles."""

    def __init__(
        self,
        client: CyclesClient | AsyncCyclesClient,
        *,
        subject: SubjectConfig,
        action: ActionConfig,
        mode: Mode = "decide",
        estimate: Amount | None = None,
        ttl_ms: int = 60_000,
        denial_message: DenialFormatter = "Tool call denied by Cycles policy: {reason}",
        settlement_error_policy: SettlementErrorPolicy = "raise",
        idempotency_namespace: IdempotencyNamespace | None = None,
        cost_fn: ToolCostFn | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}; expected one of {_VALID_MODES}.")
        if settlement_error_policy not in _VALID_SETTLEMENT_POLICIES:
            raise ValueError(
                f"Invalid settlement_error_policy {settlement_error_policy!r}; "
                f"expected one of {_VALID_SETTLEMENT_POLICIES}."
            )
        self._client = client
        self._subject = subject
        self._action = action
        self._mode = mode
        self._estimate = estimate or _DEFAULT_ESTIMATE
        self._ttl_ms = ttl_ms
        self._denial_message = denial_message
        self._settlement_error_policy: SettlementErrorPolicy = settlement_error_policy
        self._idempotency_namespace = idempotency_namespace
        self._cost_fn = cost_fn

    def _resolve_actual(self, request: Any, result: Any) -> tuple[Amount, bool]:
        """Resolve the commit ``actual`` from cost_fn or fall back to estimate.

        Return ``(amount, used_estimate)``. If ``cost_fn`` is unset, return the
        configured estimate. If it raises or returns an invalid value, log a
        warning and fall back to the estimate so a costing bug does not erase
        the tool result.
        """
        if self._cost_fn is None:
            return self._estimate, True
        try:
            actual = self._cost_fn(request, result)
        except Exception:
            logger.warning(
                "tool cost_fn raised while computing commit amount; falling back to configured estimate",
                exc_info=True,
            )
            return self._estimate, True
        if not isinstance(actual, Amount):
            logger.warning(
                "tool cost_fn returned %s instead of Amount; falling back to configured estimate",
                type(actual).__name__,
            )
            return self._estimate, True
        return actual, False

    def _resolve_namespace(self, ctx: Any) -> str | None:
        """Resolve the optional namespace for this call. Returns None if disabled."""
        if self._idempotency_namespace is None:
            return None
        if callable(self._idempotency_namespace):
            return self._idempotency_namespace(ctx)
        return self._idempotency_namespace

    # ------------------------------------------------------------------ sync

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if not isinstance(self._client, CyclesClient):
            raise TypeError(
                "CyclesToolGate.wrap_tool_call requires a sync CyclesClient. "
                "Pass an AsyncCyclesClient and run the agent via .ainvoke() to use the async path."
            )
        tool_call = get_tool_call(request)
        tool_name = str(tool_call.get("name", "")) or None
        tool_call_id = _coerce_id(tool_call.get("id"))
        subject = resolve_subject(self._subject, request, get_state(request))
        action = resolve_action(self._action, request, tool_name)
        namespace = self._resolve_namespace(request)

        if self._mode in ("decide", "decide+reserve"):
            decide_resp = self._client.decide(
                DecisionRequest(
                    idempotency_key=make_idempotency_key("decide", tool_call_id, namespace=namespace),
                    subject=subject,
                    action=action,
                    estimate=self._estimate,
                )
            )
            if not is_allowed(decide_resp):
                return ToolMessage(
                    content=format_denial(self._denial_message, decide_resp, tool_name),
                    tool_call_id=tool_call_id,
                )

        if self._mode == "decide":
            return handler(request)

        return self._reserve_and_run_sync(request, handler, subject, action, tool_name, tool_call_id, namespace)

    def _reserve_and_run_sync(
        self,
        request: Any,
        handler: Callable[[Any], Any],
        subject: Any,
        action: Any,
        tool_name: str | None,
        tool_call_id: str,
        namespace: str | None,
    ) -> Any:
        assert isinstance(self._client, CyclesClient)
        idem = make_idempotency_key("res", tool_call_id, namespace=namespace)
        commit_metadata: dict[str, Any] = {}
        reservation = self._client.stream_reservation(
            idempotency_key=idem,
            subject=subject,
            action=action,
            estimate=self._estimate,
            ttl_ms=self._ttl_ms,
            raise_on_commit_failure=self._settlement_error_policy == "raise",
            metadata=commit_metadata,
        )
        try:
            reservation.__enter__()
        except CyclesProtocolError as exc:
            return ToolMessage(
                content=format_denial(self._denial_message, response_from_protocol_error(exc), tool_name),
                tool_call_id=tool_call_id,
            )

        try:
            result = handler(request)
        except BaseException as exc:
            reservation.__exit__(type(exc), exc, exc.__traceback__)
            if reservation.release_error is not None:
                logger.warning(
                    "Cycles release returned HTTP failure for reservation %s: %s",
                    reservation.reservation_id,
                    reservation.release_error,
                )
            raise

        actual, actual_from_estimate = self._resolve_actual(request, result)
        if actual.unit != self._estimate.unit:
            logger.warning("tool cost_fn returned a different unit; falling back to configured estimate")
            actual = self._estimate
            actual_from_estimate = True
        if actual_from_estimate:
            commit_metadata["actual_source"] = "estimate"
        reservation.usage.actual_cost = actual.amount
        reservation.__exit__(None, None, None)
        if reservation.settlement_error is not None:
            if reservation.settlement_error.status < 0:
                logger.warning(
                    "Cycles commit raised for reservation %s; durable recovery is queued",
                    reservation.reservation_id,
                )
            else:
                logger.warning(
                    "Cycles commit returned HTTP failure for reservation %s; durable recovery is queued",
                    reservation.reservation_id,
                )
        return result

    # ----------------------------------------------------------------- async

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if not isinstance(self._client, AsyncCyclesClient):
            raise TypeError(
                "CyclesToolGate.awrap_tool_call requires an AsyncCyclesClient. "
                "Pass a CyclesClient and run the agent via .invoke() to use the sync path."
            )
        tool_call = get_tool_call(request)
        tool_name = str(tool_call.get("name", "")) or None
        tool_call_id = _coerce_id(tool_call.get("id"))
        subject = resolve_subject(self._subject, request, get_state(request))
        action = resolve_action(self._action, request, tool_name)
        namespace = self._resolve_namespace(request)

        if self._mode in ("decide", "decide+reserve"):
            decide_resp = await self._client.decide(
                DecisionRequest(
                    idempotency_key=make_idempotency_key("decide", tool_call_id, namespace=namespace),
                    subject=subject,
                    action=action,
                    estimate=self._estimate,
                )
            )
            if not is_allowed(decide_resp):
                return ToolMessage(
                    content=format_denial(self._denial_message, decide_resp, tool_name),
                    tool_call_id=tool_call_id,
                )

        if self._mode == "decide":
            result = handler(request)
            if hasattr(result, "__await__"):
                result = await result
            return result

        return await self._reserve_and_run_async(request, handler, subject, action, tool_name, tool_call_id, namespace)

    async def _reserve_and_run_async(
        self,
        request: Any,
        handler: Callable[[Any], Any],
        subject: Any,
        action: Any,
        tool_name: str | None,
        tool_call_id: str,
        namespace: str | None,
    ) -> Any:
        assert isinstance(self._client, AsyncCyclesClient)
        idem = make_idempotency_key("res", tool_call_id, namespace=namespace)
        commit_metadata: dict[str, Any] = {}
        reservation = self._client.stream_reservation(
            idempotency_key=idem,
            subject=subject,
            action=action,
            estimate=self._estimate,
            ttl_ms=self._ttl_ms,
            raise_on_commit_failure=self._settlement_error_policy == "raise",
            metadata=commit_metadata,
        )
        try:
            await reservation.__aenter__()
        except CyclesProtocolError as exc:
            return ToolMessage(
                content=format_denial(self._denial_message, response_from_protocol_error(exc), tool_name),
                tool_call_id=tool_call_id,
            )

        try:
            result = handler(request)
            if hasattr(result, "__await__"):
                result = await result
        except BaseException as exc:
            await reservation.__aexit__(type(exc), exc, exc.__traceback__)
            if reservation.release_error is not None:
                logger.warning(
                    "Cycles release returned HTTP failure for reservation %s: %s",
                    reservation.reservation_id,
                    reservation.release_error,
                )
            raise

        actual, actual_from_estimate = self._resolve_actual(request, result)
        if actual.unit != self._estimate.unit:
            logger.warning("tool cost_fn returned a different unit; falling back to configured estimate")
            actual = self._estimate
            actual_from_estimate = True
        if actual_from_estimate:
            commit_metadata["actual_source"] = "estimate"
        reservation.usage.actual_cost = actual.amount
        await reservation.__aexit__(None, None, None)
        if reservation.settlement_error is not None:
            if reservation.settlement_error.status < 0:
                logger.warning(
                    "Cycles commit raised for reservation %s; durable recovery is queued",
                    reservation.reservation_id,
                )
            else:
                logger.warning(
                    "Cycles commit returned HTTP failure for reservation %s; durable recovery is queued",
                    reservation.reservation_id,
                )
        return result
