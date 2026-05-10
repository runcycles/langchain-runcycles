"""CyclesFanOutGate — turn-counting middleware that halts runaway agents.

Tracks how many model turns have occurred in a single agent run and halts the
agent when a configured cap is reached. The halt is a graceful one — emit an
``AIMessage`` and ``jump_to: "end"`` so downstream code sees an expected stop,
not an exception.

Optionally consults ``client.decide()`` on each model turn so an external
policy service can reject continued fan-out (e.g. a tenant whose burst budget
is exhausted). When ``client`` is provided and decide() denies, the agent halts
with the denial reason as the AIMessage content.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage
from runcycles import Action, AsyncCyclesClient, CyclesClient, DecisionRequest

from langchain_runcycles._config import DenialFormatter, IdempotencyNamespace, SubjectConfig, TurnCounter
from langchain_runcycles._internal import (
    _DEFAULT_ESTIMATE,
    denial_reason,
    is_allowed,
    make_idempotency_key,
    resolve_action,
    resolve_subject,
)

logger = logging.getLogger(__name__)

# Fan-out doesn't gate per-tool, so a per-tool-name Mapping doesn't make sense here.
# Accepting only static Action or a state-derived callable.
FanOutActionConfig: TypeAlias = Action | Callable[[Any], Action]


def _default_turn_counter(state: Any) -> int:
    """Count AIMessages in the message history. Each one is one model turn."""
    messages = _get_messages(state)
    return sum(1 for m in messages if _is_ai_message(m))


def _get_messages(state: Any) -> list[Any]:
    if isinstance(state, dict):
        msgs = state.get("messages") or []
    else:
        msgs = getattr(state, "messages", None) or []
    return list(msgs)


def _is_ai_message(message: Any) -> bool:
    cls_name = type(message).__name__
    if cls_name == "AIMessage":
        return True
    msg_type = getattr(message, "type", None)
    return msg_type == "ai"


class CyclesFanOutGate(AgentMiddleware):
    """Halt the agent before model invocation when a turn cap or external policy says stop."""

    def __init__(
        self,
        max_turns: int,
        *,
        client: CyclesClient | AsyncCyclesClient | None = None,
        subject: SubjectConfig | None = None,
        action: FanOutActionConfig | None = None,
        turn_counter: TurnCounter = _default_turn_counter,
        cap_message: str = "Fan-out cap reached: {turns} of {max_turns} model turns used.",
        denial_message: DenialFormatter = "Agent halted by Cycles policy: {reason}",
        idempotency_namespace: IdempotencyNamespace | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if client is not None and (subject is None or action is None):
            raise ValueError("subject and action are required when a Cycles client is provided.")
        if isinstance(action, Mapping):
            raise TypeError(
                "CyclesFanOutGate.action does not support per-tool Mapping — it gates model turns, "
                "not tool calls. Pass a single Action or a Callable[[state], Action]."
            )
        self._max_turns = max_turns
        self._client = client
        self._subject = subject
        self._action = action
        self._turn_counter = turn_counter
        self._cap_message = cap_message
        self._denial_message = denial_message
        self._idempotency_namespace = idempotency_namespace

    def _resolve_namespace(self, state: Any) -> str | None:
        """Resolve the optional namespace from state. Returns None if disabled."""
        if self._idempotency_namespace is None:
            return None
        if callable(self._idempotency_namespace):
            return self._idempotency_namespace(state)
        return self._idempotency_namespace

    # ------------------------------------------------------------------ sync

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any | None = None) -> dict[str, Any] | None:
        del runtime
        turns = self._turn_counter(state)
        if turns >= self._max_turns:
            return self._halt(self._cap_message.format(turns=turns, max_turns=self._max_turns))

        if self._client is None:
            return None
        if not isinstance(self._client, CyclesClient):
            raise TypeError(
                "CyclesFanOutGate.before_model requires a sync CyclesClient. "
                "Pass an AsyncCyclesClient and run the agent via .ainvoke() to use the async path."
            )
        decide_resp = self._client.decide(
            DecisionRequest(
                idempotency_key=make_idempotency_key(
                    "fanout-decide", namespace=self._resolve_namespace(state)
                ),
                subject=resolve_subject(self._subject, state, state),
                action=resolve_action(self._action, state, None),
                estimate=_DEFAULT_ESTIMATE,
            )
        )
        if not is_allowed(decide_resp):
            reason = denial_reason(decide_resp)
            if callable(self._denial_message):
                content = self._denial_message(decide_resp)
            else:
                content = self._denial_message.format(reason=reason, tool="<model>", decision="DENY")
            return self._halt(content)
        return None

    # ----------------------------------------------------------------- async

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any | None = None) -> dict[str, Any] | None:
        del runtime
        turns = self._turn_counter(state)
        if turns >= self._max_turns:
            return self._halt(self._cap_message.format(turns=turns, max_turns=self._max_turns))

        if self._client is None:
            return None
        if not isinstance(self._client, AsyncCyclesClient):
            raise TypeError(
                "CyclesFanOutGate.abefore_model requires an AsyncCyclesClient. "
                "Pass a CyclesClient and run the agent via .invoke() to use the sync path."
            )
        decide_resp = await self._client.decide(
            DecisionRequest(
                idempotency_key=make_idempotency_key(
                    "fanout-decide", namespace=self._resolve_namespace(state)
                ),
                subject=resolve_subject(self._subject, state, state),
                action=resolve_action(self._action, state, None),
                estimate=_DEFAULT_ESTIMATE,
            )
        )
        if not is_allowed(decide_resp):
            reason = denial_reason(decide_resp)
            if callable(self._denial_message):
                content = self._denial_message(decide_resp)
            else:
                content = self._denial_message.format(reason=reason, tool="<model>", decision="DENY")
            return self._halt(content)
        return None

    @staticmethod
    def _halt(content: str) -> dict[str, Any]:
        return {"messages": [AIMessage(content=content)], "jump_to": "end"}
