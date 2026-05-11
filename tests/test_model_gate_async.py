"""Tests for CyclesModelGate async path (awrap_model_call)."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage
from runcycles import AsyncCyclesClient, CyclesClient, CyclesResponse

from langchain_runcycles import CyclesModelGate
from tests.conftest import FakeModelRequest, deny_response, reserve_failure


@pytest.mark.asyncio
async def test_async_decide_allow(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="decide")
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    result = await gate.awrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    handler.assert_called_once_with(model_request)


@pytest.mark.asyncio
async def test_async_decide_deny_returns_denial(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    async_client.decide.return_value = deny_response("OVER_QUOTA")  # type: ignore[attr-defined]
    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="decide")
    handler = MagicMock()
    result = await gate.awrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    assert "OVER_QUOTA" in result.result[0].content
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_sync_client_on_async_path_raises(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="decide")
    with pytest.raises(TypeError, match="AsyncCyclesClient"):
        await gate.awrap_model_call(model_request, lambda r: "ok")


@pytest.mark.asyncio
async def test_async_handler_returning_coroutine_is_awaited(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    async def async_handler(_r: Any) -> Any:
        return ModelResponse(result=[AIMessage(content="async result")])

    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="decide")
    result = await gate.awrap_model_call(model_request, async_handler)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "async result"


@pytest.mark.asyncio
async def test_async_reserve_lifecycle(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    result = await gate.awrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)


@pytest.mark.asyncio
async def test_async_reserve_handler_raises_releases(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    release_called: list[bool] = []

    async def fake_release(_rid: Any, _req: Any) -> CyclesResponse:
        release_called.append(True)
        return CyclesResponse.success(200, {"status": "RELEASED"})

    async_client.release_reservation = fake_release  # type: ignore[method-assign]

    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="reserve")

    def boom(_r: Any) -> Any:
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        await gate.awrap_model_call(model_request, boom)
    assert release_called == [True]


@pytest.mark.asyncio
async def test_async_reserve_failure_returns_denial(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    async def fake_create(_req: Any) -> CyclesResponse:
        return reserve_failure()

    async_client.create_reservation = fake_create  # type: ignore[method-assign]
    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock()
    result = await gate.awrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_async_settlement_raise_default(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    async def failing_commit(_rid: Any, _req: Any) -> CyclesResponse:
        raise RuntimeError("cycles unavailable")

    async_client.commit_reservation = failing_commit  # type: ignore[method-assign]
    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    with pytest.raises(RuntimeError, match="cycles unavailable"):
        await gate.awrap_model_call(model_request, handler)


@pytest.mark.asyncio
async def test_async_settlement_log(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    async def failing_commit(_rid: Any, _req: Any) -> CyclesResponse:
        raise RuntimeError("cycles unavailable")

    async_client.commit_reservation = failing_commit  # type: ignore[method-assign]
    gate = CyclesModelGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        settlement_error_policy="log",
    )
    handler_result = ModelResponse(result=[AIMessage(content="model output")])
    handler = MagicMock(return_value=handler_result)
    result = await gate.awrap_model_call(model_request, handler)
    assert result is handler_result


@pytest.mark.asyncio
async def test_async_namespace_callable_receives_request(
    async_client: AsyncCyclesClient, subject: Any, action: Any
) -> None:
    captured: dict[str, str] = {}

    async def capture_decide(req: Any) -> CyclesResponse:
        captured["key"] = req.idempotency_key
        return CyclesResponse.success(200, {"decision": "ALLOW", "affected_scopes": []})

    async_client.decide = capture_decide  # type: ignore[method-assign]

    def derive(request: Any) -> str:
        return request.state["run_id"]

    gate = CyclesModelGate(
        async_client,
        subject=subject,
        action=action,
        mode="decide",
        idempotency_namespace=derive,
    )
    request = FakeModelRequest(state={"run_id": "async_run_xyz", "messages": []})
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    await gate.awrap_model_call(request, handler)
    assert re.match(r"^model-decide-async_run_xyz-[0-9a-f]{32}$", captured["key"])


# --- cost_fn (v0.2.0+) ---------------------------------------------------------------


def _capture_commit_actual(async_client: AsyncCyclesClient, captured: dict[str, Any]) -> None:
    """Replace async_client.commit_reservation with a wrapper that records
    the CommitRequest into the captured dict before returning success."""
    from tests.conftest import commit_ok

    async def _commit(_rid: Any, request: Any) -> Any:
        captured["request"] = request
        return commit_ok()

    async_client.commit_reservation = _commit  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_async_cost_fn_used_for_commit_actual(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    """async parity: commit `actual` comes from cost_fn(result), not estimate."""
    from runcycles import Amount, Unit

    actual_from_cost_fn = Amount(unit=Unit.USD_MICROCENTS, amount=12_345)

    def cost(_result: Any) -> Amount:
        return actual_from_cost_fn

    captured: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured)

    gate = CyclesModelGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        cost_fn=cost,
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    await gate.awrap_model_call(model_request, handler)

    assert captured["request"].actual.amount == 12_345
    assert captured["request"].actual.unit == Unit.USD_MICROCENTS


@pytest.mark.asyncio
async def test_async_cost_fn_none_commits_at_estimate(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    """async parity: without cost_fn, commit `actual` equals the configured estimate."""
    from runcycles import Amount, Unit

    estimate = Amount(unit=Unit.USD_MICROCENTS, amount=99_999)
    captured: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured)

    gate = CyclesModelGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        estimate=estimate,
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    await gate.awrap_model_call(model_request, handler)

    assert captured["request"].actual.amount == 99_999


@pytest.mark.asyncio
async def test_async_cost_fn_exception_falls_back_to_estimate(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    """async parity: cost_fn that raises must not erase the model result."""
    from runcycles import Amount, Unit

    def broken_cost(_result: Any) -> Amount:
        raise ValueError("provider response shape unrecognized")

    estimate = Amount(unit=Unit.USD_MICROCENTS, amount=42)
    captured: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured)

    gate = CyclesModelGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        estimate=estimate,
        cost_fn=broken_cost,
    )
    handler_result = ModelResponse(result=[AIMessage(content="model output")])
    handler = MagicMock(return_value=handler_result)

    result = await gate.awrap_model_call(model_request, handler)
    assert result is handler_result

    assert captured["request"].actual.amount == 42


@pytest.mark.asyncio
async def test_async_cost_fn_used_in_decide_reserve_mode(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    """async parity for decide+reserve + cost_fn — the same parity guarantee as the
    sync test_cost_fn_used_in_decide_reserve_mode."""
    from runcycles import Amount, Unit

    actual_from_cost_fn = Amount(unit=Unit.USD_MICROCENTS, amount=7_777)

    def cost(_result: Any) -> Amount:
        return actual_from_cost_fn

    captured: dict[str, Any] = {}
    _capture_commit_actual(async_client, captured)

    gate = CyclesModelGate(
        async_client,
        subject=subject,
        action=action,
        mode="decide+reserve",
        cost_fn=cost,
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    await gate.awrap_model_call(model_request, handler)

    assert captured["request"].actual.amount == 7_777


@pytest.mark.asyncio
async def test_async_reservation_missing_id_returns_denial(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    """async parity for the sync `test_reservation_missing_id_returns_denial`:
    a successful create_reservation response with no `reservation_id` is treated
    as a denial — the handler must not run."""
    async def _create_missing_id(_request: Any) -> CyclesResponse:
        return CyclesResponse.success(201, {"decision": "ALLOW", "affected_scopes": []})

    async_client.create_reservation = _create_missing_id  # type: ignore[method-assign]
    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock()
    result = await gate.awrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_async_handler_returning_awaitable_in_reserve_mode(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    """The reserve-mode async path awaits awaitable handler results (line 307).
    LangChain agents pass an async handler when invoked via .ainvoke(), so this
    awaitable branch is the real production code path — not just the
    decide-mode one already covered."""
    expected = ModelResponse(result=[AIMessage(content="async ok")])

    async def async_handler(_request: Any) -> Any:
        return expected

    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="reserve")
    result = await gate.awrap_model_call(model_request, async_handler)
    assert result is expected
