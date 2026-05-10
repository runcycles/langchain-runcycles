"""Shared fixtures for the langchain-runcycles test suite."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from runcycles import (
    Action,
    AsyncCyclesClient,
    CyclesClient,
    CyclesConfig,
    CyclesResponse,
    Subject,
)


def allow_response(reservation_id: str | None = None) -> CyclesResponse:
    body: dict[str, Any] = {"decision": "ALLOW", "affected_scopes": []}
    if reservation_id is not None:
        body["reservation_id"] = reservation_id
    return CyclesResponse.success(200, body)


def deny_response(reason_code: str = "BUDGET_EXCEEDED") -> CyclesResponse:
    return CyclesResponse.success(
        200, {"decision": "DENY", "reason_code": reason_code, "affected_scopes": []}
    )


def reserve_success(reservation_id: str = "rsv_test_1") -> CyclesResponse:
    return CyclesResponse.success(
        201,
        {
            "decision": "ALLOW",
            "reservation_id": reservation_id,
            "affected_scopes": [],
        },
    )


def reserve_failure() -> CyclesResponse:
    return CyclesResponse.http_error(402, "Insufficient budget", {"error": "BUDGET_EXCEEDED", "message": "out"})


def commit_ok() -> CyclesResponse:
    return CyclesResponse.success(200, {"status": "COMMITTED", "charged": {"unit": "USD_MICROCENTS", "amount": 10000}})


def release_ok() -> CyclesResponse:
    return CyclesResponse.success(200, {"status": "RELEASED", "released": {"unit": "USD_MICROCENTS", "amount": 10000}})


@pytest.fixture
def sync_client() -> Iterator[CyclesClient]:
    config = CyclesConfig(base_url="http://test", api_key="test-key", tenant="acme")
    client = CyclesClient(config)
    client.decide = MagicMock(return_value=allow_response())  # type: ignore[method-assign]
    client.create_reservation = MagicMock(return_value=reserve_success())  # type: ignore[method-assign]
    client.commit_reservation = MagicMock(return_value=commit_ok())  # type: ignore[method-assign]
    client.release_reservation = MagicMock(return_value=release_ok())  # type: ignore[method-assign]
    yield client
    client.close()


# Note: AsyncCyclesClient cleanup intentionally omitted. The httpx.AsyncClient inside
# would need `await client.aclose()` to close cleanly, which requires running the
# event loop in fixture teardown. Since methods are mock-replaced and no real HTTP
# is issued, the underlying connection pool is empty — letting GC reclaim it is fine
# for short-lived test runs. Convert to @pytest_asyncio.fixture if real HTTP is added.
@pytest.fixture
def async_client() -> Iterator[AsyncCyclesClient]:
    config = CyclesConfig(base_url="http://test", api_key="test-key", tenant="acme")
    client = AsyncCyclesClient(config)

    async def _decide(_request: Any) -> CyclesResponse:
        return _decide.return_value  # type: ignore[attr-defined]

    async def _create(_request: Any) -> CyclesResponse:
        return _create.return_value  # type: ignore[attr-defined]

    async def _commit(_rid: Any, _request: Any) -> CyclesResponse:
        return _commit.return_value  # type: ignore[attr-defined]

    async def _release(_rid: Any, _request: Any) -> CyclesResponse:
        return _release.return_value  # type: ignore[attr-defined]

    _decide.return_value = allow_response()  # type: ignore[attr-defined]
    _create.return_value = reserve_success()  # type: ignore[attr-defined]
    _commit.return_value = commit_ok()  # type: ignore[attr-defined]
    _release.return_value = release_ok()  # type: ignore[attr-defined]

    client.decide = _decide  # type: ignore[method-assign]
    client.create_reservation = _create  # type: ignore[method-assign]
    client.commit_reservation = _commit  # type: ignore[method-assign]
    client.release_reservation = _release  # type: ignore[method-assign]
    yield client


@pytest.fixture
def subject() -> Subject:
    return Subject(tenant="acme", agent="research-bot")


@pytest.fixture
def action() -> Action:
    return Action(kind="tool.call", name="send_email")


class FakeToolCallRequest:
    """Minimal stand-in for a LangChain ToolCallRequest. Real AgentMiddleware.wrap_tool_call accepts this shape."""

    def __init__(self, name: str = "send_email", args: dict[str, Any] | None = None, call_id: str = "tc_1"):
        self.tool_call: dict[str, Any] = {"name": name, "args": args or {}, "id": call_id}
        self.state: dict[str, Any] = {"messages": []}


@pytest.fixture
def tool_call_request() -> FakeToolCallRequest:
    return FakeToolCallRequest()
