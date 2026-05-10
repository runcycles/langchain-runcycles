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
