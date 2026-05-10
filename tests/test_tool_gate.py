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


# --- Tier 2 review-driven additions -------------------------------------------------


def test_commit_called_with_configured_estimate(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """The reserve-mode commit body must carry actual=self._estimate, not some other amount.

    AUDIT.md documents that v0.1 commits at estimate; this test makes that contract
    enforceable so a future refactor that changes the commit amount must also update
    the audit and this assertion.
    """
    from runcycles import Amount, Unit

    custom_estimate = Amount(unit=Unit.USD_MICROCENTS, amount=12_345)
    gate = CyclesToolGate(
        sync_client, subject=subject, action=action, mode="reserve", estimate=custom_estimate
    )
    gate.wrap_tool_call(tool_call_request, lambda r: "ok")
    args, _ = sync_client.commit_reservation.call_args  # type: ignore[attr-defined]
    commit_request = args[1]
    assert commit_request.actual == custom_estimate


def test_idempotency_keys_are_deterministic_per_tool_call_id(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Idempotency keys are deterministic when tool_call_id is supplied.

    Shape: ``<prefix>-<tool_call_id>``. Same tool_call_id MUST produce the same
    key on retry so the Cycles server treats it as one reservation, not two.
    The parent SDK and the publish workflow both depend on this convention.
    """
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide+reserve")
    gate.wrap_tool_call(tool_call_request, lambda r: "ok")

    decide_args, _ = sync_client.decide.call_args  # type: ignore[attr-defined]
    assert decide_args[0].idempotency_key == "decide-tc_1"

    reserve_args, _ = sync_client.create_reservation.call_args  # type: ignore[attr-defined]
    reserve_key = reserve_args[0].idempotency_key
    assert reserve_key == "res-tc_1"

    commit_args, _ = sync_client.commit_reservation.call_args  # type: ignore[attr-defined]
    assert commit_args[1].idempotency_key == f"commit-{reserve_key}"


def test_idempotency_key_retry_lands_on_same_key(
    sync_client: CyclesClient, subject: Any, action: Any
) -> None:
    """Two `wrap_tool_call` invocations with the same tool_call_id produce the same key.

    This is the core retry-stability property: a duplicated dispatch (e.g. by a durable
    workflow runner replaying state) must NOT create a second reservation in Cycles.
    """
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")
    gate.wrap_tool_call(FakeToolCallRequest(call_id="abc-123"), lambda r: "ok")
    first_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    gate.wrap_tool_call(FakeToolCallRequest(call_id="abc-123"), lambda r: "ok")
    second_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    assert first_key == second_key == "decide-abc-123"


def test_synthetic_tool_call_id_when_missing(
    sync_client: CyclesClient, subject: Any, action: Any
) -> None:
    """Missing tool_call_id is replaced with `missing-<12-hex>` and a warning is logged.

    Empty strings would silently break LangChain's tool-call/response correlation.
    """
    import logging
    import re

    request = FakeToolCallRequest(call_id="")
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")

    with pytest.LogCaptureHandler() if False else _capture_warnings("langchain_runcycles._internal") as logs:
        result = gate.wrap_tool_call(request, lambda r: "should not run")

    assert isinstance(result, str) or result == "should not run"  # decide allowed; no ToolMessage
    # Trigger the deny path so the synthetic id flows into a ToolMessage and we can assert it
    sync_client.decide.return_value = deny_response("X")  # type: ignore[attr-defined]
    request2 = FakeToolCallRequest(call_id="")
    result2 = gate.wrap_tool_call(request2, lambda r: "never")
    assert isinstance(result2, ToolMessage)
    assert re.match(r"^missing-[0-9a-f]{12}$", result2.tool_call_id)
    # And a warning was emitted at least once
    warning_messages = [r for r in logs if r.levelno >= logging.WARNING]
    assert any("synthesized tool_call_id" in r.getMessage() for r in warning_messages)


def test_settlement_raise_default_propagates_commit_failure(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Strict-governance default: when commit fails after a successful tool run, raise.

    The tool already produced its side effect; silently dropping the bookkeeping
    would let unaccounted spend through. Surfacing the exception forces the caller
    to reconcile.
    """
    sync_client.commit_reservation.side_effect = RuntimeError("cycles unavailable")  # type: ignore[attr-defined]
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="reserve")
    with pytest.raises(RuntimeError, match="cycles unavailable"):
        gate.wrap_tool_call(tool_call_request, lambda r: "tool ran")
    sync_client.commit_reservation.assert_called_once()  # type: ignore[attr-defined]


def test_settlement_log_swallows_commit_failure(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Opt-in `log` policy: commit failure is logged, tool result returned.

    Matches the v0.1.0/v0.1.1 behavior for users who chose UX over strict accounting.
    """
    sync_client.commit_reservation.side_effect = RuntimeError("cycles unavailable")  # type: ignore[attr-defined]
    gate = CyclesToolGate(
        sync_client, subject=subject, action=action, mode="reserve", settlement_error_policy="log"
    )
    with _capture_warnings("langchain_runcycles.tool_gate") as logs:
        result = gate.wrap_tool_call(tool_call_request, lambda r: "tool ran")
    assert result == "tool ran"
    assert any("commit failed" in r.getMessage() for r in logs)


def test_invalid_settlement_policy_raises(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    with pytest.raises(ValueError, match="Invalid settlement_error_policy"):
        CyclesToolGate(
            sync_client, subject=subject, action=action, settlement_error_policy="bogus"  # type: ignore[arg-type]
        )


# --- v0.1.3 idempotency namespace tests -------------------------------------------


def test_make_idempotency_key_with_namespace_and_suffix() -> None:
    from langchain_runcycles._internal import make_idempotency_key

    assert make_idempotency_key("res", "tc_1", namespace="run_abc") == "res-run_abc-tc_1"


def test_make_idempotency_key_namespace_without_suffix_uses_uuid() -> None:
    """Run-scoped fan-out style: namespace present, suffix None → UUID fills the call slot."""
    import re

    from langchain_runcycles._internal import make_idempotency_key

    key = make_idempotency_key("fanout-decide", namespace="run_abc")
    assert re.match(r"^fanout-decide-run_abc-[0-9a-f]{32}$", key)


def test_make_idempotency_key_no_namespace_keeps_v012_shape() -> None:
    """v0.1.2 backward compatibility: omitting namespace gives the v0.1.2 shape unchanged."""
    from langchain_runcycles._internal import make_idempotency_key

    assert make_idempotency_key("res", "tc_1") == "res-tc_1"


def test_idempotency_namespace_as_static_string(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Static-string namespace is woven into every Cycles key for the tool gate."""
    gate = CyclesToolGate(
        sync_client,
        subject=subject,
        action=action,
        mode="decide+reserve",
        idempotency_namespace="run_abc",
    )
    gate.wrap_tool_call(tool_call_request, lambda r: "ok")

    decide_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    reserve_key = sync_client.create_reservation.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    commit_key = sync_client.commit_reservation.call_args[0][1].idempotency_key  # type: ignore[attr-defined]

    assert decide_key == "decide-run_abc-tc_1"
    assert reserve_key == "res-run_abc-tc_1"
    assert commit_key == f"commit-{reserve_key}"


def test_idempotency_namespace_as_callable(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Callable namespace is evaluated per call with the request as argument."""
    captured: list[Any] = []

    def derive(request: Any) -> str:
        captured.append(request)
        return "run_xyz"

    gate = CyclesToolGate(
        sync_client,
        subject=subject,
        action=action,
        mode="decide",
        idempotency_namespace=derive,
    )
    gate.wrap_tool_call(tool_call_request, lambda r: "ok")

    assert captured == [tool_call_request]
    decide_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    assert decide_key == "decide-run_xyz-tc_1"


def test_no_namespace_preserves_v012_key_shape(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Backward-compat regression: without idempotency_namespace, keys match v0.1.2 exactly."""
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")
    gate.wrap_tool_call(tool_call_request, lambda r: "ok")
    decide_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    assert decide_key == "decide-tc_1"


def test_namespace_prevents_cross_run_collision(
    sync_client: CyclesClient, subject: Any, action: Any
) -> None:
    """Same tool_call_id under different namespaces produces different keys.

    This is the failure mode the namespace exists to prevent: two runs that both
    use a short, framework-generated tool_call_id like 'tc_1' would otherwise
    collide on the same Cycles reservation.
    """
    gate_run_a = CyclesToolGate(
        sync_client, subject=subject, action=action, mode="decide", idempotency_namespace="run_a"
    )
    gate_run_b = CyclesToolGate(
        sync_client, subject=subject, action=action, mode="decide", idempotency_namespace="run_b"
    )
    gate_run_a.wrap_tool_call(FakeToolCallRequest(call_id="tc_1"), lambda r: "ok")
    key_a = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    gate_run_b.wrap_tool_call(FakeToolCallRequest(call_id="tc_1"), lambda r: "ok")
    key_b = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    assert key_a == "decide-run_a-tc_1"
    assert key_b == "decide-run_b-tc_1"
    assert key_a != key_b


def test_callable_namespace_returning_none_opts_out_per_call(
    sync_client: CyclesClient, subject: Any, action: Any
) -> None:
    """A callable namespace returning None disables namespacing for that call.

    Useful when some calls should be globally scoped (admin / system tools) and
    others run-scoped — the user can branch on `request.tool_call['name']` and
    return None for the unscoped path, falling back to the v0.1.2 shape.
    """
    def conditional_namespace(request: Any) -> str | None:
        if request.tool_call["name"] == "admin_tool":
            return None
        return "user_run"

    gate = CyclesToolGate(
        sync_client,
        subject=subject,
        action=action,
        mode="decide",
        idempotency_namespace=conditional_namespace,
    )
    gate.wrap_tool_call(FakeToolCallRequest(name="admin_tool", call_id="tc_admin"), lambda r: "ok")
    admin_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    gate.wrap_tool_call(FakeToolCallRequest(name="user_tool", call_id="tc_user"), lambda r: "ok")
    user_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    assert admin_key == "decide-tc_admin"  # v0.1.2 shape (no namespace)
    assert user_key == "decide-user_run-tc_user"  # namespaced


def test_release_key_inherits_namespace_on_tool_exception(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """release_reservation key inherits the namespace via the reservation key it composes from.

    Without this guarantee, a retried release call would target the wrong
    reservation (or none at all) under runs with reused tool_call_ids.
    """
    gate = CyclesToolGate(
        sync_client,
        subject=subject,
        action=action,
        mode="reserve",
        idempotency_namespace="run_abc",
    )

    def boom(_r: Any) -> Any:
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        gate.wrap_tool_call(tool_call_request, boom)

    release_args, _ = sync_client.release_reservation.call_args  # type: ignore[attr-defined]
    release_key = release_args[1].idempotency_key
    assert release_key == "release-res-run_abc-tc_1"


# --- helper: a tiny capturer scoped to a logger -----------------------------------


from contextlib import contextmanager  # noqa: E402
from logging import LogRecord  # noqa: E402


@contextmanager
def _capture_warnings(logger_name: str):  # type: ignore[no-untyped-def]
    """Capture LogRecords for a logger inside a `with` block."""
    import logging

    records: list[LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: LogRecord) -> None:
            records.append(record)

    handler = _Handler(level=logging.DEBUG)
    log = logging.getLogger(logger_name)
    log.addHandler(handler)
    previous_level = log.level
    log.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        log.removeHandler(handler)
        log.setLevel(previous_level)
