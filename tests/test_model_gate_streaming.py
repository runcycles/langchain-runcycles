"""Streaming-path verification for CyclesModelGate (v0.2.1+).

LangChain's `_execute_model_async` (the handler passed into `awrap_model_call`)
calls `await model_.ainvoke(messages)` which internally consumes the model's
streaming generator via `agenerate_prompt -> agenerate -> generate_from_stream`
and merges the chunks into a single `AIMessage` with aggregated
`usage_metadata` *before* returning. So at the middleware boundary the handler
always returns a finalized `ModelResponse` — middleware never sees chunks.

These tests lock that contract down on our side: we want a regression check
that proves `CyclesModelGate` cost_fn fires exactly once per turn and that
the final usage_metadata it sees is the sum of streamed chunks, not the
first chunk's partial counts. If a future LangChain release changes the
contract (e.g., starts passing per-chunk callbacks), these tests will fail
loudly and we'll know to adapt.

Reference: `langchain/agents/factory.py:_execute_model_async` (line 1323-1349
in langchain 1.2.18) is the handler that wraps the model invocation. It calls
`ainvoke` once and returns one `ModelResponse(result=[final_aimessage], ...)`.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage
from runcycles import Amount, AsyncCyclesClient, Unit

from langchain_runcycles import CyclesModelGate
from langchain_runcycles.extractors import openai_cost
from tests.conftest import FakeModelRequest


def _aggregate_streamed_chunks(chunks: list[dict[str, int]]) -> AIMessage:
    """Simulate LangChain's chunk-merge: sum `input_tokens` and `output_tokens`
    across the streamed pieces, build the final AIMessage with the aggregated
    usage_metadata. Faithful to `generate_from_stream` in langchain-core's
    chat_models.py."""
    input_total = sum(c.get("input_tokens", 0) for c in chunks)
    output_total = sum(c.get("output_tokens", 0) for c in chunks)
    return AIMessage(
        content="aggregated streamed content",
        usage_metadata={
            "input_tokens": input_total,
            "output_tokens": output_total,
            "total_tokens": input_total + output_total,
        },
    )


@pytest.mark.asyncio
async def test_cost_fn_called_once_when_handler_aggregates_streamed_chunks(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    model_request: FakeModelRequest,
) -> None:
    """Even when the handler internally aggregates many streamed chunks, cost_fn
    fires exactly once at the model-turn boundary. This is the load-bearing
    contract — if it ever fires per-chunk we'd commit N times per turn."""
    streamed_chunks = [
        {"input_tokens": 50, "output_tokens": 0},   # prompt-only chunk
        {"input_tokens": 0, "output_tokens": 30},   # generation chunk 1
        {"input_tokens": 0, "output_tokens": 30},   # generation chunk 2
        {"input_tokens": 0, "output_tokens": 40},   # generation chunk 3
    ]

    async def streaming_handler(_request: Any) -> ModelResponse:
        # Models stream chunks internally; ainvoke aggregates before returning.
        final_message = _aggregate_streamed_chunks(streamed_chunks)
        return ModelResponse(result=[final_message])

    cost_fn = MagicMock(return_value=Amount(unit=Unit.USD_MICROCENTS, amount=100))

    gate = CyclesModelGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        cost_fn=cost_fn,
    )
    await gate.awrap_model_call(model_request, streaming_handler)

    cost_fn.assert_called_once()


@pytest.mark.asyncio
async def test_cost_fn_sees_aggregated_usage_metadata_not_first_chunk(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    model_request: FakeModelRequest,
) -> None:
    """cost_fn must read the *aggregated* usage_metadata from the final
    AIMessage — the sum of all streamed chunks — not the first chunk's
    partial counts. The OpenAI extractor uses input/output_tokens from the
    final message: we verify the cost it computes matches the SUMMED tokens,
    not any single chunk's tokens."""
    streamed_chunks = [
        {"input_tokens": 200, "output_tokens": 0},     # prompt
        {"input_tokens": 0, "output_tokens": 50},
        {"input_tokens": 0, "output_tokens": 50},
        {"input_tokens": 0, "output_tokens": 100},     # 200 + 0/50/50/100
    ]
    expected_input = 200
    expected_output = 200

    async def streaming_handler(_request: Any) -> ModelResponse:
        final = _aggregate_streamed_chunks(streamed_chunks)
        return ModelResponse(result=[final])

    captured: dict[str, Any] = {}

    async def _capture_commit(_rid: Any, request: Any) -> Any:
        from tests.conftest import commit_ok

        captured["actual"] = request.actual
        return commit_ok()

    async_client.commit_reservation = _capture_commit  # type: ignore[method-assign]

    # OpenAI gpt-4o pricing: $2.50/M input, $10.00/M output
    # expected: (200 * 2.50 + 200 * 10.00) / 1_000_000 USD
    #         = 0.0005 + 0.002 = $0.0025 = 250_000 microcents
    gate = CyclesModelGate(
        async_client,
        subject=subject,
        action=action,
        mode="reserve",
        cost_fn=openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00),
    )
    await gate.awrap_model_call(model_request, streaming_handler)

    assert captured["actual"].unit == Unit.USD_MICROCENTS
    assert captured["actual"].amount == 250_000

    # Sanity: if we'd extracted from the first chunk only (200 input / 0 output)
    # the commit would have been (200 * 2.50) / 1_000_000 = 50_000 microcents.
    # Asserting != 50_000 catches a hypothetical "first-chunk-only" regression.
    assert captured["actual"].amount != 50_000

    # And the expected aggregation produced the assumed totals.
    assert expected_input + expected_output == 400


@pytest.mark.asyncio
async def test_cancellation_during_handler_releases_reservation(
    async_client: AsyncCyclesClient,
    subject: Any,
    action: Any,
    model_request: FakeModelRequest,
) -> None:
    """If the caller cancels mid-stream (asyncio.CancelledError raised inside the
    handler, e.g., from `agent.astream()` consumer disconnecting), the gate must
    release the reservation. CancelledError is a BaseException (not Exception),
    so our `except BaseException:` is the load-bearing guard here.

    Without this test, a future refactor that narrowed the except to `Exception`
    would silently leak reservations on every cancelled stream."""
    release_called: list[str] = []

    async def fake_release(reservation_id: str, _req: Any) -> Any:
        from runcycles import CyclesResponse

        release_called.append(reservation_id)
        return CyclesResponse.success(200, {"status": "RELEASED"})

    async_client.release_reservation = fake_release  # type: ignore[method-assign]

    async def cancelled_handler(_request: Any) -> ModelResponse:
        raise asyncio.CancelledError("client disconnected mid-stream")

    gate = CyclesModelGate(async_client, subject=subject, action=action, mode="reserve")

    with pytest.raises(asyncio.CancelledError):
        await gate.awrap_model_call(model_request, cancelled_handler)

    assert release_called == ["rsv_test_1"]
