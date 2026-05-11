"""CyclesModelGate — pre-model-call authorization middleware for LangChain agents.

Wraps every model invocation with a Cycles authorization check. Three modes
(parity with :class:`CyclesToolGate`):

  * ``"decide"`` — call ``client.decide()`` before invoking the model. On denial,
    returns a ``ModelResponse`` whose ``AIMessage`` carries the denial reason;
    the agent terminates naturally because the AIMessage has no tool_calls.
  * ``"reserve"`` — create a reservation, call the model, commit on success or
    release on exception. Used when model calls consume budget that should be
    debited up front.
  * ``"decide+reserve"`` — authorize via ``decide()`` first, then reserve and
    commit. Most strict.

v0.2.0+ supports per-call actual-cost extraction via the optional ``cost_fn``
parameter: if supplied, the callable receives the ``ModelResponse`` returned
by the wrapped handler and returns an ``Amount`` used for the commit. If the
callable raises or returns a non-``Amount``, the gate logs a warning and falls
back to the configured ``estimate`` so a costing bug never erases a successful
model result.
Without ``cost_fn`` the gate commits at the configured ``estimate`` (the
v0.1.x default behavior). ``langchain_runcycles.extractors`` ships
``openai_cost`` and ``anthropic_cost`` factories that produce a ``cost_fn``
parameterized by per-million-token pricing.

ModelGate idempotency keys use a UUID per-call slot since model requests
don't carry an upstream-stable id like ``tool_call_id`` (namespacing still
works as a run/workflow scope).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from runcycles import (
    Amount,
    AsyncCyclesClient,
    CommitRequest,
    CyclesClient,
    DecisionRequest,
    ReleaseRequest,
    ReservationCreateRequest,
)

from langchain_runcycles._config import (
    ActionConfig,
    CostFn,
    DenialFormatter,
    IdempotencyNamespace,
    SubjectConfig,
)
from langchain_runcycles._internal import (
    _DEFAULT_ESTIMATE,
    denial_reason,
    format_denial,
    get_state,
    is_allowed,
    make_idempotency_key,
    resolve_action,
    resolve_subject,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_VALID_MODES: tuple[str, ...] = ("decide", "reserve", "decide+reserve")
_VALID_SETTLEMENT_POLICIES: tuple[str, ...] = ("raise", "log")
Mode = Literal["decide", "reserve", "decide+reserve"]
SettlementErrorPolicy = Literal["raise", "log"]


class CyclesModelGate(AgentMiddleware):
    """Authorize each LangChain agent model call against Cycles."""

    def __init__(
        self,
        client: CyclesClient | AsyncCyclesClient,
        *,
        subject: SubjectConfig,
        action: ActionConfig,
        mode: Mode = "decide",
        estimate: Amount | None = None,
        ttl_ms: int = 60_000,
        denial_message: DenialFormatter = "Model call denied by Cycles policy: {reason}",
        settlement_error_policy: SettlementErrorPolicy = "raise",
        idempotency_namespace: IdempotencyNamespace | None = None,
        cost_fn: CostFn | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}; expected one of {_VALID_MODES}.")
        if settlement_error_policy not in _VALID_SETTLEMENT_POLICIES:
            raise ValueError(
                f"Invalid settlement_error_policy {settlement_error_policy!r}; "
                f"expected one of {_VALID_SETTLEMENT_POLICIES}."
            )
        if isinstance(action, Mapping):
            raise TypeError(
                "CyclesModelGate.action does not support per-tool Mapping: model "
                "calls don't carry a tool name. Pass a single Action or a "
                "Callable[[ModelRequest], Action]."
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

    def _resolve_actual(self, result: Any) -> Amount:
        """Resolve the commit ``actual`` from cost_fn or fall back to estimate.

        If ``cost_fn`` is unset, return the configured estimate (v0.1.x parity).
        If it's set and returns a valid Amount, return its result. If it raises
        or returns an invalid value, log a warning and fall back to the estimate
        so a costing bug does not erase the model result."""
        if self._cost_fn is None:
            return self._estimate
        try:
            actual = self._cost_fn(result)
        except Exception:
            logger.warning(
                "cost_fn raised while computing commit amount; falling back to configured estimate",
                exc_info=True,
            )
            return self._estimate
        if not isinstance(actual, Amount):
            logger.warning(
                "cost_fn returned %s instead of Amount; falling back to configured estimate",
                type(actual).__name__,
            )
            return self._estimate
        return actual

    def _resolve_namespace(self, ctx: Any) -> str | None:
        """Resolve the optional namespace for this call. Returns None if disabled."""
        if self._idempotency_namespace is None:
            return None
        if callable(self._idempotency_namespace):
            return self._idempotency_namespace(ctx)
        return self._idempotency_namespace

    def _denied_response(self, decide_resp: Any) -> Any:
        """Build a ModelResponse whose AIMessage carries the denial reason.

        Imported lazily so test code that doesn't exercise the denial path
        doesn't pay the langchain.agents.middleware import cost.
        """
        from langchain.agents.middleware import ModelResponse

        msg = AIMessage(content=format_denial(self._denial_message, decide_resp, "<model>"))
        return ModelResponse(result=[msg])

    # ------------------------------------------------------------------ sync

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if not isinstance(self._client, CyclesClient):
            raise TypeError(
                "CyclesModelGate.wrap_model_call requires a sync CyclesClient. "
                "Pass an AsyncCyclesClient and run the agent via .ainvoke() to use the async path."
            )
        subject = resolve_subject(self._subject, request, get_state(request))
        action = resolve_action(self._action, request, None)
        namespace = self._resolve_namespace(request)

        if self._mode in ("decide", "decide+reserve"):
            decide_resp = self._client.decide(
                DecisionRequest(
                    idempotency_key=make_idempotency_key("model-decide", namespace=namespace),
                    subject=subject,
                    action=action,
                    estimate=self._estimate,
                )
            )
            if not is_allowed(decide_resp):
                return self._denied_response(decide_resp)

        if self._mode == "decide":
            return handler(request)

        return self._reserve_and_run_sync(request, handler, subject, action, namespace)

    def _reserve_and_run_sync(
        self,
        request: Any,
        handler: Callable[[Any], Any],
        subject: Any,
        action: Any,
        namespace: str | None,
    ) -> Any:
        assert isinstance(self._client, CyclesClient)
        idem = make_idempotency_key("model-res", namespace=namespace)
        reserve_resp = self._client.create_reservation(
            ReservationCreateRequest(
                idempotency_key=idem,
                subject=subject,
                action=action,
                estimate=self._estimate,
                ttl_ms=self._ttl_ms,
            )
        )
        if not reserve_resp.is_success:
            return self._denied_response(reserve_resp)
        reservation_id = reserve_resp.get_body_attribute("reservation_id")
        if not reservation_id:
            return self._denied_response(reserve_resp)

        try:
            result = handler(request)
        except BaseException:
            # Same intent as CyclesToolGate: release on KeyboardInterrupt /
            # SystemExit / asyncio.CancelledError as well as ordinary exceptions,
            # then re-raise so the original cause wins.
            self._safe_release_sync(reservation_id, idem)
            raise

        actual = self._resolve_actual(result)
        try:
            commit_resp = self._client.commit_reservation(
                reservation_id,
                CommitRequest(
                    idempotency_key=f"commit-{idem}",
                    actual=actual,
                ),
            )
        except Exception:
            if self._settlement_error_policy == "raise":
                raise
            logger.warning(
                "Cycles commit raised for model reservation %s; settlement_error_policy='log' "
                "so the model result is preserved (reservation will expire via TTL)",
                reservation_id,
                exc_info=True,
            )
            return result
        if not commit_resp.is_success:
            if self._settlement_error_policy == "raise":
                raise RuntimeError(
                    f"Cycles commit_reservation returned HTTP failure for model reservation "
                    f"{reservation_id}: {denial_reason(commit_resp)}"
                )
            logger.warning(
                "Cycles commit returned HTTP failure for model reservation %s: %s; "
                "settlement_error_policy='log' so the model result is preserved",
                reservation_id,
                denial_reason(commit_resp),
            )
        return result

    def _safe_release_sync(self, reservation_id: str, idem: str) -> None:
        assert isinstance(self._client, CyclesClient)
        try:
            release_resp = self._client.release_reservation(
                reservation_id,
                ReleaseRequest(idempotency_key=f"release-{idem}"),
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Cycles release raised for reservation %s", reservation_id, exc_info=True)
            return
        if not release_resp.is_success:
            logger.warning(
                "Cycles release returned HTTP failure for reservation %s: %s",
                reservation_id,
                denial_reason(release_resp),
            )

    # ----------------------------------------------------------------- async

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if not isinstance(self._client, AsyncCyclesClient):
            raise TypeError(
                "CyclesModelGate.awrap_model_call requires an AsyncCyclesClient. "
                "Pass a CyclesClient and run the agent via .invoke() to use the sync path."
            )
        subject = resolve_subject(self._subject, request, get_state(request))
        action = resolve_action(self._action, request, None)
        namespace = self._resolve_namespace(request)

        if self._mode in ("decide", "decide+reserve"):
            decide_resp = await self._client.decide(
                DecisionRequest(
                    idempotency_key=make_idempotency_key("model-decide", namespace=namespace),
                    subject=subject,
                    action=action,
                    estimate=self._estimate,
                )
            )
            if not is_allowed(decide_resp):
                return self._denied_response(decide_resp)

        if self._mode == "decide":
            result = handler(request)
            if hasattr(result, "__await__"):
                result = await result
            return result

        return await self._reserve_and_run_async(request, handler, subject, action, namespace)

    async def _reserve_and_run_async(
        self,
        request: Any,
        handler: Callable[[Any], Any],
        subject: Any,
        action: Any,
        namespace: str | None,
    ) -> Any:
        assert isinstance(self._client, AsyncCyclesClient)
        idem = make_idempotency_key("model-res", namespace=namespace)
        reserve_resp = await self._client.create_reservation(
            ReservationCreateRequest(
                idempotency_key=idem,
                subject=subject,
                action=action,
                estimate=self._estimate,
                ttl_ms=self._ttl_ms,
            )
        )
        if not reserve_resp.is_success:
            return self._denied_response(reserve_resp)
        reservation_id = reserve_resp.get_body_attribute("reservation_id")
        if not reservation_id:
            return self._denied_response(reserve_resp)

        try:
            result = handler(request)
            if hasattr(result, "__await__"):
                result = await result
        except BaseException:
            await self._safe_release_async(reservation_id, idem)
            raise

        actual = self._resolve_actual(result)
        try:
            commit_resp = await self._client.commit_reservation(
                reservation_id,
                CommitRequest(
                    idempotency_key=f"commit-{idem}",
                    actual=actual,
                ),
            )
        except Exception:
            if self._settlement_error_policy == "raise":
                raise
            logger.warning(
                "Cycles commit raised for model reservation %s; settlement_error_policy='log' "
                "so the model result is preserved (reservation will expire via TTL)",
                reservation_id,
                exc_info=True,
            )
            return result
        if not commit_resp.is_success:
            if self._settlement_error_policy == "raise":
                raise RuntimeError(
                    f"Cycles commit_reservation returned HTTP failure for model reservation "
                    f"{reservation_id}: {denial_reason(commit_resp)}"
                )
            logger.warning(
                "Cycles commit returned HTTP failure for model reservation %s: %s; "
                "settlement_error_policy='log' so the model result is preserved",
                reservation_id,
                denial_reason(commit_resp),
            )
        return result

    async def _safe_release_async(self, reservation_id: str, idem: str) -> None:
        assert isinstance(self._client, AsyncCyclesClient)
        try:
            release_resp = await self._client.release_reservation(
                reservation_id,
                ReleaseRequest(idempotency_key=f"release-{idem}"),
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Cycles release raised for reservation %s", reservation_id, exc_info=True)
            return
        if not release_resp.is_success:
            logger.warning(
                "Cycles release returned HTTP failure for reservation %s: %s",
                reservation_id,
                denial_reason(release_resp),
            )
