"""Shared fixtures for the langchain-runcycles test suite."""

from __future__ import annotations

# Filter upstream import-time warnings before any test module imports happen.
# pyproject.toml's [tool.pytest.ini_options].filterwarnings activates AFTER
# collection-time imports, which is too late for warnings fired at langgraph's
# module top level (langgraph.checkpoint.serde.encrypted instantiates
# JsonPlusSerializer at class-definition time as a default argument, which
# triggers langchain-core's deprecation machinery).
#
# Filtering by class object is more reliable than by message regex here
# because langchain-core's deprecation decorator builds the warning via
# warnings.warn(instance, category=ParentClass, stacklevel=4) which makes the
# message-regex match brittle.
import warnings  # noqa: E402

from langchain_core._api.deprecation import (  # noqa: E402
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
)

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.1[3-9]",
)

from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
from runcycles import (  # noqa: E402
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
        200,
        {
            "decision": "ALLOW",
            "reservation_id": reservation_id,
            "expires_at_ms": 9_000_000_000_000,
            "remaining_ttl_ms": 60_000,
            "affected_scopes": [],
        },
    )


def reserve_failure() -> CyclesResponse:
    return CyclesResponse.http_error(402, "Insufficient budget", {"error": "BUDGET_EXCEEDED", "message": "out"})


def commit_ok() -> CyclesResponse:
    return CyclesResponse.success(200, {"status": "COMMITTED", "charged": {"unit": "USD_MICROCENTS", "amount": 10000}})


def commit_failure() -> CyclesResponse:
    """HTTP-failure commit response (e.g., 5xx from Cycles server, transient outage).

    Used by v0.2.3+ regression tests for the settlement_error_policy contract on
    non-success commit responses (previously the gate silently treated HTTP
    failures as success)."""
    return CyclesResponse.http_error(
        503, "Service unavailable", {"error": "SERVICE_UNAVAILABLE", "message": "downstream timeout"}
    )


def release_failure() -> CyclesResponse:
    """HTTP-failure release response. Best-effort cleanup; never raises, always logs."""
    return CyclesResponse.http_error(
        500, "Internal server error", {"error": "INTERNAL", "message": "release path failed"}
    )


def release_ok() -> CyclesResponse:
    return CyclesResponse.success(200, {"status": "RELEASED", "released": {"unit": "USD_MICROCENTS", "amount": 10000}})


@pytest.fixture
def sync_client(tmp_path: Path) -> Iterator[CyclesClient]:
    config = CyclesConfig(
        base_url="http://test",
        api_key="test-key",
        tenant="acme",
        journal_dir=str(tmp_path / "journal"),
        retry_enabled=False,
    )
    client = CyclesClient(config)
    client.decide = MagicMock(return_value=allow_response())  # type: ignore[method-assign]
    client.create_reservation = MagicMock(return_value=reserve_success())  # type: ignore[method-assign]
    client.commit_reservation = MagicMock(return_value=commit_ok())  # type: ignore[method-assign]
    client.release_reservation = MagicMock(return_value=release_ok())  # type: ignore[method-assign]
    client.extend_reservation = MagicMock()  # type: ignore[method-assign]
    yield client
    client.close()


# Note: AsyncCyclesClient cleanup intentionally omitted. The httpx.AsyncClient inside
# would need `await client.aclose()` to close cleanly, which requires running the
# event loop in fixture teardown. Since methods are mock-replaced and no real HTTP
# is issued, the underlying connection pool is empty — letting GC reclaim it is fine
# for short-lived test runs. Convert to @pytest_asyncio.fixture if real HTTP is added.
@pytest.fixture
def async_client(tmp_path: Path) -> Iterator[AsyncCyclesClient]:
    config = CyclesConfig(
        base_url="http://test",
        api_key="test-key",
        tenant="acme",
        journal_dir=str(tmp_path / "journal"),
        retry_enabled=False,
    )
    client = AsyncCyclesClient(config)

    async def _decide(_request: Any) -> CyclesResponse:
        return _decide.return_value  # type: ignore[attr-defined]

    async def _create(_request: Any) -> CyclesResponse:
        return _create.return_value  # type: ignore[attr-defined]

    async def _commit(_rid: Any, _request: Any) -> CyclesResponse:
        return _commit.return_value  # type: ignore[attr-defined]

    async def _release(_rid: Any, _request: Any) -> CyclesResponse:
        return _release.return_value  # type: ignore[attr-defined]

    async def _extend(_rid: Any, _request: Any) -> CyclesResponse:
        return _extend.return_value  # type: ignore[attr-defined]

    _decide.return_value = allow_response()  # type: ignore[attr-defined]
    _create.return_value = reserve_success()  # type: ignore[attr-defined]
    _commit.return_value = commit_ok()  # type: ignore[attr-defined]
    _release.return_value = release_ok()  # type: ignore[attr-defined]
    _extend.return_value = CyclesResponse.success(
        200,
        {"status": "EXTENDED", "expires_at_ms": 9_000_000_060_000, "remaining_ttl_ms": 60_000},
    )  # type: ignore[attr-defined]

    client.decide = _decide  # type: ignore[method-assign]
    client.create_reservation = _create  # type: ignore[method-assign]
    client.commit_reservation = _commit  # type: ignore[method-assign]
    client.release_reservation = _release  # type: ignore[method-assign]
    client.extend_reservation = _extend  # type: ignore[method-assign]
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


class FakeModelRequest:
    """Minimal stand-in for a LangChain ModelRequest.

    Real ``ModelRequest`` is a dataclass with required fields like ``model``
    (a ``BaseChatModel``) and ``runtime`` (a ``Runtime``). The middleware only
    reads ``request.state`` (via ``get_state``); other fields are passed
    through to the handler unchanged. So a duck-typed object is enough for
    middleware-level tests.
    """

    def __init__(self, state: dict[str, Any] | None = None):
        self.state: dict[str, Any] = state if state is not None else {"messages": []}


@pytest.fixture
def model_request() -> FakeModelRequest:
    return FakeModelRequest()
