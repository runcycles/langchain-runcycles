"""Tests for CyclesToolGate sync path."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from runcycles import Action, AsyncCyclesClient, CyclesClient

from langchain_runcycles import CyclesToolGate
from langchain_runcycles._internal import (
    denial_reason,
    format_denial,
    get_tool_call,
    is_allowed,
    parse_decision,
    resolve_action,
    resolve_subject,
)
from tests.conftest import (
    FakeToolCallRequest,
    deny_response,
    reserve_failure,
)


def test_invalid_mode_raises(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    with pytest.raises(ValueError, match="Invalid mode"):
        CyclesToolGate(sync_client, subject=subject, action=action, mode="bogus")  # type: ignore[arg-type]


def test_decide_allow_invokes_handler(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")
    handler = MagicMock(return_value="tool result")
    result = gate.wrap_tool_call(tool_call_request, handler)
    assert result == "tool result"
    handler.assert_called_once_with(tool_call_request)
    sync_client.decide.assert_called_once()  # type: ignore[attr-defined]


def test_decide_deny_returns_tool_message(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    sync_client.decide.return_value = deny_response("BUDGET_EXCEEDED")  # type: ignore[attr-defined]
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")
    handler = MagicMock()
    result = gate.wrap_tool_call(tool_call_request, handler)
    assert isinstance(result, ToolMessage)
    assert "BUDGET_EXCEEDED" in result.content
    assert result.tool_call_id == "tc_1"
    handler.assert_not_called()


def test_async_client_on_sync_path_raises(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="decide")
    with pytest.raises(TypeError, match="sync CyclesClient"):
        gate.wrap_tool_call(tool_call_request, lambda r: "ok")


def test_reserve_mode_full_lifecycle(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="reserve")
    result = gate.wrap_tool_call(tool_call_request, lambda r: "tool ok")
    assert result == "tool ok"
    sync_client.create_reservation.assert_called_once()  # type: ignore[attr-defined]
    sync_client.commit_reservation.assert_called_once()  # type: ignore[attr-defined]
    sync_client.release_reservation.assert_not_called()  # type: ignore[attr-defined]
    args, _ = sync_client.commit_reservation.call_args  # type: ignore[attr-defined]
    assert args[0] == "rsv_test_1"


def test_reserve_handler_raises_releases(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="reserve")

    def boom(_r: Any) -> Any:
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        gate.wrap_tool_call(tool_call_request, boom)
    sync_client.release_reservation.assert_called_once()  # type: ignore[attr-defined]
    sync_client.commit_reservation.assert_not_called()  # type: ignore[attr-defined]


def test_reserve_failure_returns_tool_message(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    sync_client.create_reservation.return_value = reserve_failure()  # type: ignore[attr-defined]
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock()
    result = gate.wrap_tool_call(tool_call_request, handler)
    assert isinstance(result, ToolMessage)
    handler.assert_not_called()


def test_decide_then_reserve_short_circuits_on_deny(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    sync_client.decide.return_value = deny_response()  # type: ignore[attr-defined]
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide+reserve")
    handler = MagicMock()
    result = gate.wrap_tool_call(tool_call_request, handler)
    assert isinstance(result, ToolMessage)
    sync_client.create_reservation.assert_not_called()  # type: ignore[attr-defined]
    handler.assert_not_called()


def test_decide_then_reserve_full_path(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide+reserve")
    result = gate.wrap_tool_call(tool_call_request, lambda r: "ok")
    assert result == "ok"
    sync_client.decide.assert_called_once()  # type: ignore[attr-defined]
    sync_client.create_reservation.assert_called_once()  # type: ignore[attr-defined]
    sync_client.commit_reservation.assert_called_once()  # type: ignore[attr-defined]


def test_reservation_missing_id_returns_denial(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    from runcycles import CyclesResponse

    sync_client.create_reservation.return_value = CyclesResponse.success(  # type: ignore[attr-defined]
        201, {"decision": "ALLOW", "affected_scopes": []}
    )
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock()
    result = gate.wrap_tool_call(tool_call_request, handler)
    assert isinstance(result, ToolMessage)
    handler.assert_not_called()


def test_action_mapping_resolves_per_tool(sync_client: CyclesClient, subject: Any) -> None:
    mapping = {
        "send_email": Action(kind="tool.call", name="send_email"),
        "search": Action(kind="tool.call", name="search"),
    }
    gate = CyclesToolGate(sync_client, subject=subject, action=mapping, mode="decide")
    gate.wrap_tool_call(FakeToolCallRequest(name="search", call_id="tc_2"), lambda r: "ok")
    args, _ = sync_client.decide.call_args  # type: ignore[attr-defined]
    assert args[0].action.name == "search"


def test_action_mapping_missing_tool_raises(sync_client: CyclesClient, subject: Any) -> None:
    mapping = {"send_email": Action(kind="tool.call", name="send_email")}
    gate = CyclesToolGate(sync_client, subject=subject, action=mapping, mode="decide")
    with pytest.raises(KeyError, match="No action mapping"):
        gate.wrap_tool_call(FakeToolCallRequest(name="unknown_tool"), lambda r: "ok")


def test_action_callable(sync_client: CyclesClient, subject: Any) -> None:
    def derive(req: Any) -> Action:
        return Action(kind="tool.call", name=f"derived-{req.tool_call['name']}")

    gate = CyclesToolGate(sync_client, subject=subject, action=derive, mode="decide")
    gate.wrap_tool_call(FakeToolCallRequest(name="x"), lambda r: "ok")
    args, _ = sync_client.decide.call_args  # type: ignore[attr-defined]
    assert args[0].action.name == "derived-x"


def test_subject_extractor(sync_client: CyclesClient, action: Any) -> None:
    from runcycles import Subject

    def extract(_request: Any, _state: Any) -> Subject:
        return Subject(tenant="dynamic-tenant", agent="extracted")

    gate = CyclesToolGate(sync_client, subject=extract, action=action, mode="decide")
    gate.wrap_tool_call(FakeToolCallRequest(), lambda r: "ok")
    args, _ = sync_client.decide.call_args  # type: ignore[attr-defined]
    assert args[0].subject.tenant == "dynamic-tenant"


def test_denial_message_callable(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    sync_client.decide.return_value = deny_response("CUSTOM_REASON")  # type: ignore[attr-defined]
    gate = CyclesToolGate(
        sync_client,
        subject=subject,
        action=action,
        mode="decide",
        denial_message=lambda response: f"custom: {response.body['reason_code']}",
    )
    result = gate.wrap_tool_call(tool_call_request, lambda r: "never")
    assert isinstance(result, ToolMessage)
    assert result.content == "custom: CUSTOM_REASON"


def test_get_tool_call_dict_input() -> None:
    request = {"tool_call": {"name": "x", "args": {}, "id": "abc"}}
    assert get_tool_call(request)["name"] == "x"


def test_get_tool_call_missing_raises() -> None:
    with pytest.raises(AttributeError, match="missing a 'tool_call'"):
        get_tool_call(object())


def test_resolve_action_with_static_action() -> None:
    a = Action(kind="tool.call", name="x")
    assert resolve_action(a, None, None) is a


def test_resolve_action_mapping_no_tool_name() -> None:
    mapping = {"x": Action(kind="tool.call", name="x")}
    with pytest.raises(ValueError, match="requires a tool name"):
        resolve_action(mapping, None, None)


def test_resolve_subject_static() -> None:
    from runcycles import Subject

    s = Subject(tenant="t")
    assert resolve_subject(s, None, None) is s


def test_parse_decision_malformed() -> None:
    from runcycles import CyclesResponse

    bad = CyclesResponse.success(200, {"unrelated": "junk"})
    assert parse_decision(bad) is None
    assert is_allowed(bad) is False


def test_parse_decision_failure_path() -> None:
    from runcycles import CyclesResponse

    transport_err = CyclesResponse.transport_error(RuntimeError("network"))
    assert parse_decision(transport_err) is None
    assert is_allowed(transport_err) is False


def test_denial_reason_falls_back_to_error_message() -> None:
    from runcycles import CyclesResponse

    resp = CyclesResponse.http_error(500, "Server error")
    assert denial_reason(resp) == "Server error"


def test_denial_reason_default() -> None:
    from runcycles import CyclesResponse

    resp = CyclesResponse(status=599)
    assert denial_reason(resp) == "denied"


def test_format_denial_with_default_string() -> None:
    formatter = "tool={tool} reason={reason} decision={decision}"
    msg = format_denial(formatter, deny_response("BAD"), "send_email")
    assert msg == "tool=send_email reason=BAD decision=DENY"


def test_format_denial_default_decision_when_unparseable() -> None:
    from runcycles import CyclesResponse

    resp = CyclesResponse.http_error(500, "boom")
    msg = format_denial("decision={decision}", resp, None)
    assert msg == "decision=DENY"


def test_get_tool_call_attribute_path() -> None:
    class R:
        tool_call = {"name": "x", "args": {}, "id": "abc"}

    assert get_tool_call(R())["name"] == "x"
