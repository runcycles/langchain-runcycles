"""Tests for langchain_runcycles.extractors (v0.2.0+)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage
from runcycles import Unit

from langchain_runcycles.extractors import anthropic_cost, openai_cost


def _model_response_with_usage(
    input_tokens: int,
    output_tokens: int,
    input_token_details: dict[str, int] | None = None,
) -> ModelResponse:
    """Build a ModelResponse whose first AIMessage carries LangChain's normalized
    usage_metadata shape. Both providers normalize to input/output naming via the
    LangChain integration packages."""
    msg = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            **({"input_token_details": input_token_details} if input_token_details else {}),
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


def test_openai_cost_prices_cached_input_separately() -> None:
    cost_fn = openai_cost(
        prompt_per_million_usd=2.50,
        cached_prompt_per_million_usd=1.25,
        completion_per_million_usd=10.00,
    )
    response = _model_response_with_usage(
        input_tokens=1_000,
        output_tokens=0,
        input_token_details={"cache_read": 400},
    )
    # 600 ordinary input + 400 cached input = $0.002 = 200,000 microcents.
    assert cost_fn(response).amount == 200_000


def test_anthropic_cost_prices_cache_read_and_creation_separately() -> None:
    cost_fn = anthropic_cost(
        input_per_million_usd=3.00,
        output_per_million_usd=15.00,
        cache_read_per_million_usd=0.30,
        cache_creation_per_million_usd=3.75,
    )
    response = _model_response_with_usage(
        input_tokens=1_000,
        output_tokens=0,
        input_token_details={"cache_read": 200, "cache_creation": 300},
    )
    # 500 ordinary + 200 cache reads + 300 cache writes = $0.002685.
    assert cost_fn(response).amount == 268_500


def test_anthropic_cache_creation_tiers_are_priced_independently() -> None:
    cost_fn = anthropic_cost(
        input_per_million_usd=3.00,
        output_per_million_usd=15.00,
        cache_read_per_million_usd=0.30,
        cache_creation_5m_per_million_usd=3.75,
        cache_creation_1h_per_million_usd=6.00,
    )
    response = _model_response_with_usage(
        input_tokens=1_000,
        output_tokens=100,
        input_token_details={
            "cache_read": 100,
            "cache_creation": 600,
            "ephemeral_5m_input_tokens": 200,
            "ephemeral_1h_input_tokens": 300,
        },
    )

    # 300 ordinary @ $3/M + 100 read @ $0.30/M + 100 unclassified write @
    # $3/M + 200 5m write @ $3.75/M + 300 1h write @ $6/M + 100 output @ $15/M.
    assert cost_fn(response).amount == 528_000


def test_anthropic_cache_creation_tiers_cannot_exceed_creation_total() -> None:
    cost_fn = anthropic_cost(input_per_million_usd=3.00, output_per_million_usd=15.00)
    response = _model_response_with_usage(
        input_tokens=1_000,
        output_tokens=0,
        input_token_details={
            "cache_creation": 100,
            "ephemeral_5m_input_tokens": 75,
            "ephemeral_1h_input_tokens": 50,
        },
    )

    with pytest.raises(ValueError, match="tier counts"):
        cost_fn(response)


@pytest.mark.parametrize("rate", [-1.0, float("inf"), float("nan")])
def test_invalid_pricing_rate_is_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="finite, non-negative"):
        openai_cost(prompt_per_million_usd=rate, completion_per_million_usd=10.0)


def test_cache_details_cannot_exceed_total_input() -> None:
    cost_fn = openai_cost(
        prompt_per_million_usd=2.50,
        cached_prompt_per_million_usd=1.25,
        completion_per_million_usd=10.00,
    )
    response = _model_response_with_usage(
        input_tokens=10,
        output_tokens=0,
        input_token_details={"cache_read": 11},
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        cost_fn(response)


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


def test_missing_token_fields_raise_so_gate_falls_back() -> None:
    """If the normalized usage dict is present but missing required token keys,
    raise instead of silently treating them as zero."""
    cost_fn = openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00)
    response = SimpleNamespace(result=[SimpleNamespace(usage_metadata={"total_tokens": 10})])
    with pytest.raises(ValueError, match="input_tokens"):
        cost_fn(response)


def test_negative_token_counts_raise_so_gate_falls_back() -> None:
    """Provider token counts should never be negative; raise so the gate can
    fall back to estimate instead of committing an invalid debit."""
    cost_fn = anthropic_cost(input_per_million_usd=3.00, output_per_million_usd=15.00)
    response = SimpleNamespace(
        result=[SimpleNamespace(usage_metadata={"input_tokens": -1, "output_tokens": 0})]
    )
    with pytest.raises(ValueError, match="non-negative"):
        cost_fn(response)


def test_non_integer_token_values_raise_so_gate_falls_back() -> None:
    """A non-coercible token value (string that isn't a number, None, etc.) raises
    with the underlying TypeError/ValueError chained via `from exc`. Locks down the
    int() coercion guard so a malformed provider response can't slip through and
    crash the commit path with a less-informative error."""
    cost_fn = openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00)
    response = SimpleNamespace(
        result=[SimpleNamespace(usage_metadata={"input_tokens": "abc", "output_tokens": 0})]
    )
    with pytest.raises(ValueError, match="not an integer token count") as exc_info:
        cost_fn(response)
    assert exc_info.value.__cause__ is not None  # `raise ... from exc` chain preserved


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
