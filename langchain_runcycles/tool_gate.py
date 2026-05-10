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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from runcycles import (
    Amount,
    AsyncCyclesClient,
    CommitRequest,
    CyclesClient,
    DecisionRequest,
    ReleaseRequest,
    ReservationCreateRequest,
)

from langchain_runcycles._config import ActionConfig, DenialFormatter, IdempotencyNamespace, SubjectConfig
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
)

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
        tool_call_id = coerce_tool_call_id(str(tool_call.get("id", "")) or None)
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
            return ToolMessage(
                content=format_denial(self._denial_message, reserve_resp, tool_name),
                tool_call_id=tool_call_id,
            )
        reservation_id = reserve_resp.get_body_attribute("reservation_id")
        if not reservation_id:
            return ToolMessage(
                content=format_denial(self._denial_message, reserve_resp, tool_name),
                tool_call_id=tool_call_id,
            )

        try:
            result = handler(request)
        except BaseException:
            # Catch BaseException (not just Exception) so KeyboardInterrupt / SystemExit
            # also release the reservation rather than leaking it. We re-raise immediately;
            # _safe_release_sync swallows release-side failures so the original cause wins.
            self._safe_release_sync(reservation_id, idem)
            raise

        try:
            self._client.commit_reservation(
                reservation_id,
                CommitRequest(
                    idempotency_key=f"commit-{idem}",
                    actual=self._estimate,
                ),
            )
        except Exception:
            if self._settlement_error_policy == "raise":
                # Strict governance default: tool ran but commit failed; surface the
                # exception so the caller can reconcile rather than silently dropping
                # accounting. Use settlement_error_policy="log" for best-effort UX.
                raise
            logger.warning(
                "Cycles commit failed for reservation %s; settlement_error_policy='log' "
                "so the tool result is preserved (reservation will expire via TTL)",
                reservation_id,
                exc_info=True,
            )
        return result

    def _safe_release_sync(self, reservation_id: str, idem: str) -> None:
        assert isinstance(self._client, CyclesClient)
        try:
            self._client.release_reservation(
                reservation_id,
                ReleaseRequest(idempotency_key=f"release-{idem}"),
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Cycles release failed for reservation %s", reservation_id, exc_info=True)

    # ----------------------------------------------------------------- async

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if not isinstance(self._client, AsyncCyclesClient):
            raise TypeError(
                "CyclesToolGate.awrap_tool_call requires an AsyncCyclesClient. "
                "Pass a CyclesClient and run the agent via .invoke() to use the sync path."
            )
        tool_call = get_tool_call(request)
        tool_name = str(tool_call.get("name", "")) or None
        tool_call_id = coerce_tool_call_id(str(tool_call.get("id", "")) or None)
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
            return ToolMessage(
                content=format_denial(self._denial_message, reserve_resp, tool_name),
                tool_call_id=tool_call_id,
            )
        reservation_id = reserve_resp.get_body_attribute("reservation_id")
        if not reservation_id:
            return ToolMessage(
                content=format_denial(self._denial_message, reserve_resp, tool_name),
                tool_call_id=tool_call_id,
            )

        try:
            result = handler(request)
            if hasattr(result, "__await__"):
                result = await result
        except BaseException:
            # See sync sibling: BaseException is intentional. On asyncio.CancelledError
            # the await of release may itself surface the cancellation; the original
            # cause still wins because we re-raise after best-effort release.
            await self._safe_release_async(reservation_id, idem)
            raise

        try:
            await self._client.commit_reservation(
                reservation_id,
                CommitRequest(
                    idempotency_key=f"commit-{idem}",
                    actual=self._estimate,
                ),
            )
        except Exception:
            if self._settlement_error_policy == "raise":
                # See sync sibling: governance-first default surfaces commit failure.
                raise
            logger.warning(
                "Cycles commit failed for reservation %s; settlement_error_policy='log' "
                "so the tool result is preserved (reservation will expire via TTL)",
                reservation_id,
                exc_info=True,
            )
        return result

    async def _safe_release_async(self, reservation_id: str, idem: str) -> None:
        assert isinstance(self._client, AsyncCyclesClient)
        try:
            await self._client.release_reservation(
                reservation_id,
                ReleaseRequest(idempotency_key=f"release-{idem}"),
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Cycles release failed for reservation %s", reservation_id, exc_info=True)
