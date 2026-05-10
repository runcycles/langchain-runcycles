"""Tests for CyclesFanOutGate async path (abefore_model)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from runcycles import AsyncCyclesClient, CyclesClient

from langchain_runcycles import CyclesFanOutGate
from tests.conftest import deny_response


@pytest.mark.asyncio
async def test_async_below_cap_with_client_returns_none(
    async_client: AsyncCyclesClient, subject: Any, action: Any
) -> None:
    gate = CyclesFanOutGate(5, client=async_client, subject=subject, action=action)
    state = {"messages": [AIMessage(content="t1")]}
    assert await gate.abefore_model(state) is None


@pytest.mark.asyncio
async def test_async_below_cap_no_client_returns_none(subject: Any, action: Any) -> None:
    gate = CyclesFanOutGate(5)
    state = {"messages": [AIMessage(content="t1")]}
    assert await gate.abefore_model(state) is None


@pytest.mark.asyncio
async def test_async_at_cap_halts(async_client: AsyncCyclesClient, subject: Any, action: Any) -> None:
    gate = CyclesFanOutGate(2, client=async_client, subject=subject, action=action)
    state = {"messages": [AIMessage(content="t1"), AIMessage(content="t2")]}
    result = await gate.abefore_model(state)
    assert result is not None
    assert result["jump_to"] == "end"


@pytest.mark.asyncio
async def test_async_decide_deny_halts(
    async_client: AsyncCyclesClient, subject: Any, action: Any
) -> None:
    async def deny(_request: Any) -> Any:
        return deny_response("FANOUT_DENY")

    async_client.decide = deny  # type: ignore[method-assign]
    gate = CyclesFanOutGate(5, client=async_client, subject=subject, action=action)
    state = {"messages": [AIMessage(content="t1")]}
    result = await gate.abefore_model(state)
    assert result is not None
    assert "FANOUT_DENY" in result["messages"][0].content


@pytest.mark.asyncio
async def test_sync_client_on_async_path_raises(
    sync_client: CyclesClient, subject: Any, action: Any
) -> None:
    gate = CyclesFanOutGate(5, client=sync_client, subject=subject, action=action)
    state = {"messages": [AIMessage(content="t1")]}
    with pytest.raises(TypeError, match="AsyncCyclesClient"):
        await gate.abefore_model(state)


@pytest.mark.asyncio
async def test_async_denial_message_callable(
    async_client: AsyncCyclesClient, subject: Any, action: Any
) -> None:
    async def deny(_request: Any) -> Any:
        return deny_response("X")

    async_client.decide = deny  # type: ignore[method-assign]
    gate = CyclesFanOutGate(
        5,
        client=async_client,
        subject=subject,
        action=action,
        denial_message=lambda r: f"halted: {r.body['reason_code']}",
    )
    state = {"messages": [AIMessage(content="t1")]}
    result = await gate.abefore_model(state)
    assert result is not None
    assert result["messages"][0].content == "halted: X"
