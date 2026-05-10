"""Tests for CyclesModelGate sync path."""

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

# --- construction validation ---------------------------------------------------------


def test_invalid_mode_raises(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    with pytest.raises(ValueError, match="Invalid mode"):
        CyclesModelGate(sync_client, subject=subject, action=action, mode="bogus")  # type: ignore[arg-type]


def test_invalid_settlement_policy_raises(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    with pytest.raises(ValueError, match="Invalid settlement_error_policy"):
        CyclesModelGate(
            sync_client,
            subject=subject,
            action=action,
            settlement_error_policy="bogus",  # type: ignore[arg-type]
        )


# --- decide-mode paths ---------------------------------------------------------------


def test_decide_allow_invokes_handler(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="decide")
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="model said hi")]))
    result = gate.wrap_model_call(model_request, handler)
    handler.assert_called_once_with(model_request)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "model said hi"
    sync_client.decide.assert_called_once()  # type: ignore[attr-defined]


def test_decide_deny_returns_model_response_with_denial(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    sync_client.decide.return_value = deny_response("BUDGET_EXCEEDED")  # type: ignore[attr-defined]
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="decide")
    handler = MagicMock()
    result = gate.wrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    assert "BUDGET_EXCEEDED" in result.result[0].content
    handler.assert_not_called()


def test_async_client_on_sync_path_raises(
    async_client: AsyncCyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="decide")
    with pytest.raises(TypeError, match="sync CyclesClient"):
        gate.wrap_model_call(model_request, lambda r: "ok")


# --- reserve-mode lifecycle ----------------------------------------------------------


def test_reserve_mode_full_lifecycle(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    result = gate.wrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    sync_client.create_reservation.assert_called_once()  # type: ignore[attr-defined]
    sync_client.commit_reservation.assert_called_once()  # type: ignore[attr-defined]
    sync_client.release_reservation.assert_not_called()  # type: ignore[attr-defined]
    args, _ = sync_client.commit_reservation.call_args  # type: ignore[attr-defined]
    assert args[0] == "rsv_test_1"


def test_reserve_handler_raises_releases(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="reserve")

    def boom(_r: Any) -> Any:
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        gate.wrap_model_call(model_request, boom)
    sync_client.release_reservation.assert_called_once()  # type: ignore[attr-defined]
    sync_client.commit_reservation.assert_not_called()  # type: ignore[attr-defined]


def test_reserve_failure_returns_denial_response(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    sync_client.create_reservation.return_value = reserve_failure()  # type: ignore[attr-defined]
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock()
    result = gate.wrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    handler.assert_not_called()


def test_reservation_missing_id_returns_denial(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    sync_client.create_reservation.return_value = CyclesResponse.success(  # type: ignore[attr-defined]
        201, {"decision": "ALLOW", "affected_scopes": []}
    )
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock()
    result = gate.wrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    handler.assert_not_called()


# --- decide+reserve ------------------------------------------------------------------


def test_decide_then_reserve_short_circuits_on_deny(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    sync_client.decide.return_value = deny_response()  # type: ignore[attr-defined]
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="decide+reserve")
    handler = MagicMock()
    result = gate.wrap_model_call(model_request, handler)
    assert isinstance(result, ModelResponse)
    sync_client.create_reservation.assert_not_called()  # type: ignore[attr-defined]
    handler.assert_not_called()


def test_decide_then_reserve_full_path(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="decide+reserve")
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    gate.wrap_model_call(model_request, handler)
    sync_client.decide.assert_called_once()  # type: ignore[attr-defined]
    sync_client.create_reservation.assert_called_once()  # type: ignore[attr-defined]
    sync_client.commit_reservation.assert_called_once()  # type: ignore[attr-defined]


# --- settlement_error_policy ---------------------------------------------------------


def test_settlement_raise_default_propagates_commit_failure(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    sync_client.commit_reservation.side_effect = RuntimeError("cycles unavailable")  # type: ignore[attr-defined]
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="reserve")
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    with pytest.raises(RuntimeError, match="cycles unavailable"):
        gate.wrap_model_call(model_request, handler)
    sync_client.commit_reservation.assert_called_once()  # type: ignore[attr-defined]


def test_settlement_log_swallows_commit_failure(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    sync_client.commit_reservation.side_effect = RuntimeError("cycles unavailable")  # type: ignore[attr-defined]
    gate = CyclesModelGate(
        sync_client,
        subject=subject,
        action=action,
        mode="reserve",
        settlement_error_policy="log",
    )
    handler_result = ModelResponse(result=[AIMessage(content="model output")])
    handler = MagicMock(return_value=handler_result)
    result = gate.wrap_model_call(model_request, handler)
    assert result is handler_result


# --- idempotency_namespace -----------------------------------------------------------


def test_idempotency_namespace_static(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(
        sync_client,
        subject=subject,
        action=action,
        mode="decide+reserve",
        idempotency_namespace="run_abc",
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    gate.wrap_model_call(model_request, handler)

    decide_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    reserve_key = sync_client.create_reservation.call_args[0][0].idempotency_key  # type: ignore[attr-defined]

    # Model-call keys use a UUID for the per-call slot (no upstream-stable id like
    # tool_call_id available); namespace still scopes them by run.
    assert re.match(r"^model-decide-run_abc-[0-9a-f]{32}$", decide_key)
    assert re.match(r"^model-res-run_abc-[0-9a-f]{32}$", reserve_key)


def test_idempotency_namespace_callable_receives_request(
    sync_client: CyclesClient, subject: Any, action: Any
) -> None:
    captured: list[Any] = []

    def derive(request: Any) -> str:
        captured.append(request)
        return request.state["run_id"]

    gate = CyclesModelGate(
        sync_client,
        subject=subject,
        action=action,
        mode="decide",
        idempotency_namespace=derive,
    )
    request = FakeModelRequest(state={"run_id": "run_xyz", "messages": []})
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    gate.wrap_model_call(request, handler)
    assert captured == [request]
    decide_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    assert re.match(r"^model-decide-run_xyz-[0-9a-f]{32}$", decide_key)


def test_no_namespace_uses_uuid_only(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    """Without a namespace, model-call keys are pure-UUID per call (parity with v0.1.2 fanout shape)."""
    gate = CyclesModelGate(sync_client, subject=subject, action=action, mode="decide")
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    gate.wrap_model_call(model_request, handler)
    decide_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    assert re.match(r"^model-decide-[0-9a-f]{32}$", decide_key)


def test_callable_namespace_returning_none_opts_out_per_call(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    """Callable returning None disables namespacing for that call (per-call opt-out)."""
    gate = CyclesModelGate(
        sync_client,
        subject=subject,
        action=action,
        mode="decide",
        idempotency_namespace=lambda _r: None,
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="ok")]))
    gate.wrap_model_call(model_request, handler)
    decide_key = sync_client.decide.call_args[0][0].idempotency_key  # type: ignore[attr-defined]
    assert re.match(r"^model-decide-[0-9a-f]{32}$", decide_key)


def test_release_key_inherits_namespace_on_handler_exception(
    sync_client: CyclesClient, subject: Any, action: Any, model_request: FakeModelRequest
) -> None:
    gate = CyclesModelGate(
        sync_client,
        subject=subject,
        action=action,
        mode="reserve",
        idempotency_namespace="run_abc",
    )

    def boom(_r: Any) -> Any:
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        gate.wrap_model_call(model_request, boom)

    release_args, _ = sync_client.release_reservation.call_args  # type: ignore[attr-defined]
    release_key = release_args[1].idempotency_key
    # release-{idem} where idem itself starts with model-res-{namespace}-
    assert re.match(r"^release-model-res-run_abc-[0-9a-f]{32}$", release_key)
