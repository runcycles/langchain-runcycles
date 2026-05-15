"""Tests for CyclesToolGate async path (awrap_tool_call)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from runcycles import AsyncCyclesClient, CyclesClient, CyclesResponse

from langchain_runcycles import CyclesToolGate
from tests.conftest import FakeToolCallRequest, allow_response, deny_response, reserve_failure


@pytest.mark.asyncio
async def test_async_decide_allow(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="decide")
    handler = MagicMock(return_value="tool result")
    result = await gate.awrap_tool_call(tool_call_request, handler)
    assert result == "tool result"
    handler.assert_called_once_with(tool_call_request)


@pytest.mark.asyncio
async def test_async_decide_deny(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    async_client.decide.return_value = deny_response("BUDGET_EXCEEDED")  # type: ignore[attr-defined]
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="decide")
    handler = MagicMock()
    result = await gate.awrap_tool_call(tool_call_request, handler)
    assert isinstance(result, ToolMessage)
    assert "BUDGET_EXCEEDED" in result.content
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_sync_client_on_async_path_raises(
    sync_client: CyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")
    with pytest.raises(TypeError, match="AsyncCyclesClient"):
        await gate.awrap_tool_call(tool_call_request, lambda r: "ok")


@pytest.mark.asyncio
async def test_async_handler_returning_coroutine_is_awaited(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    async def async_handler(_r: Any) -> str:
        return "async tool"

    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="decide")
    result = await gate.awrap_tool_call(tool_call_request, async_handler)
    assert result == "async tool"


@pytest.mark.asyncio
async def test_async_reserve_lifecycle(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="reserve")
    result = await gate.awrap_tool_call(tool_call_request, lambda r: "ok")
    assert result == "ok"


@pytest.mark.asyncio
async def test_async_reserve_handler_raises_releases(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    release_called: list[bool] = []

    async def fake_release(_rid: Any, _req: Any) -> CyclesResponse:
        release_called.append(True)
        return allow_response()

    async_client.release_reservation = fake_release  # type: ignore[method-assign]

    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="reserve")

    def boom(_r: Any) -> Any:
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        await gate.awrap_tool_call(tool_call_request, boom)
    assert release_called == [True]


@pytest.mark.asyncio
async def test_async_reserve_failure_returns_tool_message(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    async def fake_create(_req: Any) -> CyclesResponse:
        return reserve_failure()

    async_client.create_reservation = fake_create  # type: ignore[method-assign]
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock()
    result = await gate.awrap_tool_call(tool_call_request, handler)
    assert isinstance(result, ToolMessage)
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_async_reserve_missing_reservation_id_returns_denial(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    async def fake_create(_req: Any) -> CyclesResponse:
        return CyclesResponse.success(201, {"decision": "ALLOW", "affected_scopes": []})

    async_client.create_reservation = fake_create  # type: ignore[method-assign]
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="reserve")
    result = await gate.awrap_tool_call(tool_call_request, lambda r: "never")
    assert isinstance(result, ToolMessage)


@pytest.mark.asyncio
async def test_async_decide_then_reserve_full_path(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="decide+reserve")
    result = await gate.awrap_tool_call(tool_call_request, lambda r: "ok")
    assert result == "ok"


@pytest.mark.asyncio
async def test_async_reserve_with_async_handler(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Cover the reserve-mode + coroutine-returning-handler path.

    Without this, line 247 of tool_gate.py (`result = await result` inside
    `_reserve_and_run_async`) is never exercised — the existing async coroutine
    test only covers the decide-mode branch.
    """

    async def async_handler(_r: Any) -> str:
        return "async tool result"

    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="reserve")
    result = await gate.awrap_tool_call(tool_call_request, async_handler)
    assert result == "async tool result"


@pytest.mark.asyncio
async def test_async_settlement_raise_default_propagates_commit_failure(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Strict-governance default applies on the async path too."""

    async def failing_commit(_rid: Any, _req: Any) -> CyclesResponse:
        raise RuntimeError("cycles unavailable")

    async_client.commit_reservation = failing_commit  # type: ignore[method-assign]
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="reserve")
    with pytest.raises(RuntimeError, match="cycles unavailable"):
        await gate.awrap_tool_call(tool_call_request, lambda r: "tool ran")


@pytest.mark.asyncio
async def test_async_settlement_log_swallows_commit_failure(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Async opt-in `log` policy mirrors the sync one."""

    async def failing_commit(_rid: Any, _req: Any) -> CyclesResponse:
        raise RuntimeError("cycles unavailable")

    async_client.commit_reservation = failing_commit  # type: ignore[method-assign]
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        settlement_error_policy="log",
    )
    result = await gate.awrap_tool_call(tool_call_request, lambda r: "tool ran")
    assert result == "tool ran"


@pytest.mark.asyncio
async def test_async_idempotency_namespace_static(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Static namespace flows through awrap_tool_call too."""
    captured: dict[str, str] = {}

    async def capture_decide(req: Any) -> CyclesResponse:
        captured["decide"] = req.idempotency_key
        return CyclesResponse.success(200, {"decision": "ALLOW", "affected_scopes": []})

    async_client.decide = capture_decide  # type: ignore[method-assign]
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="decide",
        idempotency_namespace="run_async_1",
    )
    await gate.awrap_tool_call(tool_call_request, lambda r: "ok")
    assert captured["decide"] == "decide-run_async_1-tc_1"


@pytest.mark.asyncio
async def test_async_idempotency_namespace_callable(
    async_client: AsyncCyclesClient, subject: Any, action: Any, tool_call_request: FakeToolCallRequest
) -> None:
    """Callable namespace evaluated with the request on the async path."""
    captured: dict[str, str] = {}

    async def capture_decide(req: Any) -> CyclesResponse:
        captured["decide"] = req.idempotency_key
        return CyclesResponse.success(200, {"decision": "ALLOW", "affected_scopes": []})

    async_client.decide = capture_decide  # type: ignore[method-assign]

    def derive_namespace(request: Any) -> str:
        return f"derived-{request.tool_call['name']}"

    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="decide",
        idempotency_namespace=derive_namespace,
    )
    await gate.awrap_tool_call(tool_call_request, lambda r: "ok")
    assert captured["decide"] == "decide-derived-send_email-tc_1"


# --- tool cost_fn (v0.3.0+) ----------------------------------------------------------


def _capture_commit_actual(async_client: AsyncCyclesClient, captured: dict[str, Any]) -> None:
    """Replace async_client.commit_reservation with a wrapper that records the CommitRequest."""
    from tests.conftest import commit_ok

    async def _commit(_rid: Any, request: Any) -> CyclesResponse:
        captured["request"] = request
        return commit_ok()

    async_client.commit_reservation = _commit


@pytest.mark.asyncio
async def test_async_tool_cost_fn_used_for_commit_actual_and_receives_request_result(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
) -> None:
    """async parity: commit actual comes from cost_fn(request, result)."""
    from runcycles import Amount, Unit

    actual_from_cost_fn = Amount(unit=Unit.USD_MICROCENTS, amount=12_345)
    captured_calls: list[tuple[Any, Any]] = []

    def cost(request: Any, result: Any) -> Amount:
        captured_calls.append((request, result))
        return actual_from_cost_fn

    captured_commit: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured_commit)
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        cost_fn=cost,
    )
    handler_result = "tool ok"
    result = await gate.awrap_tool_call(tool_call_request, lambda r: handler_result)

    assert result == handler_result
    assert captured_calls == [(tool_call_request, handler_result)]
    assert captured_commit["request"].actual.amount == 12_345
    assert captured_commit["request"].actual.unit == Unit.USD_MICROCENTS


@pytest.mark.asyncio
async def test_async_tool_cost_fn_none_commits_at_estimate(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
) -> None:
    """async parity: without cost_fn, commit actual equals the configured estimate."""
    from runcycles import Amount, Unit

    estimate = Amount(unit=Unit.USD_MICROCENTS, amount=99_999)
    captured_commit: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured_commit)
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        estimate=estimate,
    )
    await gate.awrap_tool_call(tool_call_request, lambda r: "ok")

    assert captured_commit["request"].actual == estimate


@pytest.mark.asyncio
async def test_async_tool_cost_fn_exception_falls_back_to_estimate(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """async parity: a raising cost_fn preserves the tool result and commits estimate."""
    import logging

    from runcycles import Amount, Unit

    def broken_cost(_request: Any, _result: Any) -> Amount:
        raise ValueError("tool response shape unrecognized")

    estimate = Amount(unit=Unit.USD_MICROCENTS, amount=42)
    captured_commit: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured_commit)
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        estimate=estimate,
        cost_fn=broken_cost,
    )

    with caplog.at_level(logging.WARNING, logger="langchain_runcycles.tool_gate"):
        result = await gate.awrap_tool_call(tool_call_request, lambda r: "tool output")

    assert result == "tool output"
    assert captured_commit["request"].actual == estimate
    assert "tool cost_fn raised" in caplog.text


@pytest.mark.asyncio
async def test_async_tool_cost_fn_invalid_return_falls_back_to_estimate(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """async parity: invalid cost_fn return values fall back to estimate."""
    import logging

    from runcycles import Amount, Unit

    def broken_cost(_request: Any, _result: Any) -> Any:
        return None

    estimate = Amount(unit=Unit.USD_MICROCENTS, amount=42)
    captured_commit: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured_commit)
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        estimate=estimate,
        cost_fn=broken_cost,
    )

    with caplog.at_level(logging.WARNING, logger="langchain_runcycles.tool_gate"):
        result = await gate.awrap_tool_call(tool_call_request, lambda r: "tool output")

    assert result == "tool output"
    assert captured_commit["request"].actual == estimate
    assert "tool cost_fn returned NoneType instead of Amount" in caplog.text


@pytest.mark.asyncio
async def test_async_tool_cost_fn_not_called_in_decide_mode(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
) -> None:
    """async parity: decide mode has no commit path, so cost_fn is skipped."""
    cost = MagicMock()
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="decide",
        cost_fn=cost,
    )
    result = await gate.awrap_tool_call(tool_call_request, lambda r: "tool ok")

    assert result == "tool ok"
    cost.assert_not_called()


@pytest.mark.asyncio
async def test_async_tool_cost_fn_used_in_decide_reserve_mode(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
) -> None:
    """async parity: decide+reserve reaches the same cost_fn-driven commit path."""
    from runcycles import Amount, Unit

    actual_from_cost_fn = Amount(unit=Unit.USD_MICROCENTS, amount=7_777)

    def cost(_request: Any, _result: Any) -> Amount:
        return actual_from_cost_fn

    captured_commit: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured_commit)
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="decide+reserve",
        cost_fn=cost,
    )
    await gate.awrap_tool_call(tool_call_request, lambda r: "ok")

    assert captured_commit["request"].actual.amount == 7_777


# --- v0.2.3 HTTP-failure handling on settlement paths ----------------------------------


@pytest.mark.asyncio
async def test_async_commit_http_failure_raise_default_propagates(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
) -> None:
    """async parity: non-success commit response triggers settlement_error_policy."""
    from tests.conftest import commit_failure

    async def failing_commit(_rid: Any, _req: Any) -> CyclesResponse:
        return commit_failure()

    async_client.commit_reservation = failing_commit  # type: ignore[method-assign]
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="reserve")
    with pytest.raises(RuntimeError, match="HTTP failure"):
        await gate.awrap_tool_call(tool_call_request, lambda r: "tool ran")


@pytest.mark.asyncio
async def test_async_commit_http_failure_log_swallows(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """async parity: log policy swallows HTTP-failure commits."""
    import logging

    from tests.conftest import commit_failure

    async def failing_commit(_rid: Any, _req: Any) -> CyclesResponse:
        return commit_failure()

    async_client.commit_reservation = failing_commit  # type: ignore[method-assign]
    gate = CyclesToolGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        settlement_error_policy="log",
    )
    with caplog.at_level(logging.WARNING, logger="langchain_runcycles.tool_gate"):
        result = await gate.awrap_tool_call(tool_call_request, lambda r: "tool ran")
    assert result == "tool ran"
    assert "commit returned HTTP failure" in caplog.text


@pytest.mark.asyncio
async def test_async_release_http_failure_logged(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    tool_call_request: FakeToolCallRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """async parity: non-success release is logged, never raised."""
    import logging

    from tests.conftest import release_failure

    async def failing_release(_rid: Any, _req: Any) -> CyclesResponse:
        return release_failure()

    async_client.release_reservation = failing_release  # type: ignore[method-assign]
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="reserve")

    def boom(_r: Any) -> Any:
        raise RuntimeError("tool failed")

    with caplog.at_level(logging.WARNING, logger="langchain_runcycles.tool_gate"):
        with pytest.raises(RuntimeError, match="tool failed"):
            await gate.awrap_tool_call(tool_call_request, boom)
    assert "release returned HTTP failure" in caplog.text


@pytest.mark.asyncio
async def test_async_explicit_none_tool_call_id_uses_synthetic_path(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
) -> None:
    """async parity for the sync LOW#1 fix: id=None goes through synthetic path."""
    import re

    async def deny_decide(_req: Any) -> CyclesResponse:
        return deny_response("X")

    async_client.decide = deny_decide  # type: ignore[method-assign]
    request = FakeToolCallRequest(call_id=None)  # type: ignore[arg-type]
    gate = CyclesToolGate(async_client, subject=subject, action=action, mode="decide")
    result = await gate.awrap_tool_call(request, lambda r: "never")
    assert isinstance(result, ToolMessage)
    assert re.match(r"^missing-[0-9a-f]{12}$", result.tool_call_id)
    assert result.tool_call_id != "None"
