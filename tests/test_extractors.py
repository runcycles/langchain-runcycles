"""Tests for langchain_runcycles.extractors (v0.2.0+)."""

from __future__ import annotations

import pytest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage
from runcycles import Unit

from langchain_runcycles.extractors import anthropic_cost, openai_cost


def _model_response_with_usage(input_tokens: int, output_tokens: int) -> ModelResponse:
    """Build a ModelResponse whose first AIMessage carries LangChain's normalized
    usage_metadata shape. Both providers normalize to input/output naming via the
    LangChain integration packages."""
    msg = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return ModelResponse(result=[msg])


def test_openai_cost_computes_correct_microcents() -> None:
    """OpenAI gpt-4o pricing (2026-05): $2.50/M input, $10.00/M output.
    1000 input + 500 output tokens = 0.0025 + 0.005 = $0.0075 = 750_000 microcents."""
    cost_fn = openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00)
    response = _model_response_with_usage(input_tokens=1000, output_tokens=500)
    amount = cost_fn(response)
    assert amount.unit == Unit.USD_MICROCENTS
    assert amount.amount == 750_000


def test_anthropic_cost_computes_correct_microcents() -> None:
    """Anthropic claude-sonnet-4-6 pricing (2026-05): $3.00/M input, $15.00/M output.
    2000 input + 1000 output = 0.006 + 0.015 = $0.021 = 2_100_000 microcents."""
    cost_fn = anthropic_cost(input_per_million_usd=3.00, output_per_million_usd=15.00)
    response = _model_response_with_usage(input_tokens=2000, output_tokens=1000)
    amount = cost_fn(response)
    assert amount.unit == Unit.USD_MICROCENTS
    assert amount.amount == 2_100_000


def test_zero_tokens_yields_zero_microcents() -> None:
    """Edge case: a model call that produced no output tokens should still
    successfully commit at zero. Important because providers occasionally
    report 0 output tokens on tool-only completions."""
    cost_fn = openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00)
    response = _model_response_with_usage(input_tokens=0, output_tokens=0)
    assert cost_fn(response).amount == 0


def test_missing_usage_metadata_raises_so_gate_falls_back() -> None:
    """If usage_metadata is absent from the AIMessage, the extractor raises —
    not silently returns zero — so CyclesModelGate's exception-fallback path
    debits the configured estimate (the documented contract)."""
    cost_fn = openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00)
    msg = AIMessage(content="ok")  # no usage_metadata
    response = ModelResponse(result=[msg])
    with pytest.raises(ValueError, match="usage_metadata"):
        cost_fn(response)


def test_empty_result_raises_so_gate_falls_back() -> None:
    """An empty ModelResponse.result list is unexpected but possible; raise so the
    fallback-to-estimate path covers it."""
    cost_fn = anthropic_cost(input_per_million_usd=3.00, output_per_million_usd=15.00)
    response = ModelResponse(result=[])
    with pytest.raises(ValueError, match="empty"):
        cost_fn(response)


def test_keyword_only_pricing_args() -> None:
    """Positional pricing is rejected so callers can't accidentally swap input
    and output rates (which differ ~4x on most providers, so a swap quietly
    over- or under-charges by ~4x)."""
    with pytest.raises(TypeError):
        openai_cost(2.50, 10.00)  # type: ignore[misc, call-arg]
    with pytest.raises(TypeError):
        anthropic_cost(3.00, 15.00)  # type: ignore[misc, call-arg]


def test_fractional_cent_rounding() -> None:
    """Sub-microcent fractional costs are rounded to nearest microcent, not floored
    (so total commits don't systematically under-debit accounts over many calls).
    100 input tokens at $2.50/M = $0.00025 = 25_000 microcents exactly — pick a
    pricing/token combo that exercises rounding instead."""
    cost_fn = openai_cost(prompt_per_million_usd=1.23, completion_per_million_usd=4.56)
    # 100 input * 1.23/M + 50 output * 4.56/M = 0.000123 + 0.000228 = $0.000351
    # 0.000351 USD * 100_000_000 microcents/USD = 35_100 microcents
    response = _model_response_with_usage(input_tokens=100, output_tokens=50)
    assert cost_fn(response).amount == 35_100
