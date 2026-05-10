"""Tests for CyclesFanOutGate sync path."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from runcycles import AsyncCyclesClient, CyclesClient

from langchain_runcycles import CyclesFanOutGate
from langchain_runcycles.fanout import _default_turn_counter, _is_ai_message
from tests.conftest import allow_response, deny_response


def test_max_turns_must_be_positive(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    with pytest.raises(ValueError, match="max_turns"):
        CyclesFanOutGate(0, client=sync_client, subject=subject, action=action)


def test_client_requires_subject_and_action(sync_client: CyclesClient) -> None:
    with pytest.raises(ValueError, match="subject and action"):
        CyclesFanOutGate(5, client=sync_client)


def test_below_cap_returns_none_no_decide(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    gate = CyclesFanOutGate(5, client=sync_client, subject=subject, action=action)
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}
    result = gate.before_model(state)
    assert result is None
    sync_client.decide.assert_called_once()  # type: ignore[attr-defined]


def test_below_cap_no_client_skips_decide(subject: Any, action: Any) -> None:
    gate = CyclesFanOutGate(5)
    state = {"messages": [AIMessage(content="hi")]}
    assert gate.before_model(state) is None


def test_at_cap_halts_before_decide(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    gate = CyclesFanOutGate(2, client=sync_client, subject=subject, action=action)
    state = {"messages": [AIMessage(content="t1"), AIMessage(content="t2")]}
    result = gate.before_model(state)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "Fan-out cap" in result["messages"][0].content
    sync_client.decide.assert_not_called()  # type: ignore[attr-defined]


def test_decide_deny_halts(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    sync_client.decide.return_value = deny_response("OVER_QUOTA")  # type: ignore[attr-defined]
    gate = CyclesFanOutGate(5, client=sync_client, subject=subject, action=action)
    state = {"messages": [AIMessage(content="t1")]}
    result = gate.before_model(state)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "OVER_QUOTA" in result["messages"][0].content


def test_async_client_on_sync_path_raises(
    async_client: AsyncCyclesClient, subject: Any, action: Any
) -> None:
    gate = CyclesFanOutGate(5, client=async_client, subject=subject, action=action)
    state = {"messages": [HumanMessage(content="hi")]}
    with pytest.raises(TypeError, match="sync CyclesClient"):
        gate.before_model(state)


def test_default_turn_counter_counts_ai_messages_only() -> None:
    state = {
        "messages": [
            HumanMessage(content="user1"),
            AIMessage(content="ai1"),
            HumanMessage(content="user2"),
            AIMessage(content="ai2"),
            AIMessage(content="ai3"),
        ]
    }
    assert _default_turn_counter(state) == 3


def test_turn_counter_handles_attr_state() -> None:
    class State:
        messages = [AIMessage(content="x")]

    assert _default_turn_counter(State()) == 1


def test_turn_counter_empty_state() -> None:
    assert _default_turn_counter({}) == 0
    assert _default_turn_counter(object()) == 0


def test_is_ai_message_via_type_attr() -> None:
    class FakeMessage:
        type = "ai"

    assert _is_ai_message(FakeMessage()) is True


def test_denial_message_callable(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    sync_client.decide.return_value = deny_response("X")  # type: ignore[attr-defined]
    gate = CyclesFanOutGate(
        5,
        client=sync_client,
        subject=subject,
        action=action,
        denial_message=lambda response: f"halted: {response.body['reason_code']}",
    )
    state = {"messages": [AIMessage(content="t1")]}
    result = gate.before_model(state)
    assert result is not None
    assert result["messages"][0].content == "halted: X"


def test_custom_turn_counter(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    def count_messages(state: Any) -> int:
        return len(state.get("messages", []))

    gate = CyclesFanOutGate(
        2,
        client=sync_client,
        subject=subject,
        action=action,
        turn_counter=count_messages,
    )
    state = {"messages": [HumanMessage(content="a"), HumanMessage(content="b")]}
    result = gate.before_model(state)
    assert result is not None
    assert result["jump_to"] == "end"


def test_runtime_arg_is_ignored(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    gate = CyclesFanOutGate(5, client=sync_client, subject=subject, action=action)
    state = {"messages": []}
    result = gate.before_model(state, runtime=object())
    assert result is None


def test_decide_allow_below_cap(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    sync_client.decide.return_value = allow_response()  # type: ignore[attr-defined]
    gate = CyclesFanOutGate(5, client=sync_client, subject=subject, action=action)
    state = {"messages": [AIMessage(content="hi")]}
    assert gate.before_model(state) is None


def test_fanout_rejects_mapping_action(sync_client: CyclesClient, subject: Any) -> None:
    """A per-tool-name Mapping makes no sense for fan-out (which gates model turns,
    not tool calls). Reject at construction with a clear error rather than at the
    first model turn."""
    from runcycles import Action

    mapping = {"some_tool": Action(kind="tool.call", name="some_tool")}
    with pytest.raises(TypeError, match="does not support per-tool Mapping"):
        CyclesFanOutGate(
            5,
            client=sync_client,
            subject=subject,
            action=mapping,  # type: ignore[arg-type]
        )
