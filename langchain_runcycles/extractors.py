"""Per-call cost extractors for :class:`CyclesModelGate` (v0.2.0+).

These factory functions return a ``cost_fn`` (:data:`~langchain_runcycles._config.CostFn`)
that reads token-usage metadata off the ``ModelResponse`` returned by the model
handler and converts it to a Cycles :class:`runcycles.Amount` using
caller-supplied per-million-token pricing. Pass the result as
``cost_fn=...`` to ``CyclesModelGate`` and reserve-mode commits use actual
cost instead of the configured estimate.

LangChain's ``AIMessage.usage_metadata`` is the unified surface across
providers — both ``openai_cost`` and ``anthropic_cost`` read from it. We
prefer it over provider-specific ``response_metadata`` paths because it
survives provider migrations and SDK version drift.

Pricing is per-million tokens in USD. The returned :class:`Amount` uses
:attr:`runcycles.Unit.USD_MICROCENTS` (10⁻⁸ USD) to preserve fractional-cent
precision across small calls.

Example::

    from langchain_runcycles import CyclesModelGate
    from langchain_runcycles.extractors import openai_cost

    gate = CyclesModelGate(
        client,
        subject=Subject(tenant="acme"),
        action=Action(kind="llm.completion", name="gpt-4o"),
        mode="reserve",
        estimate=Amount(unit=Unit.USD_MICROCENTS, amount=2_000_000),  # worst-case
        cost_fn=openai_cost(prompt_per_million_usd=2.50, completion_per_million_usd=10.00),
    )
"""

from __future__ import annotations

import math
from typing import Any

from runcycles import Amount, Unit

from langchain_runcycles._config import CostFn

# 1 USD = 100 cents = 100_000_000 micro-cents (10⁻⁸ USD per micro-cent).
_USD_TO_MICROCENTS = 100_000_000


def _extract_usage(result: Any) -> dict[str, Any]:
    """Pull ``usage_metadata`` off the first ``AIMessage`` in a ``ModelResponse``.

    Raises ``ValueError`` if the shape isn't recognized so the caller's
    ``CyclesModelGate`` falls back to the configured estimate (the gate
    catches every cost_fn exception by design)."""
    messages = getattr(result, "result", None)
    if not messages:
        raise ValueError("ModelResponse.result is empty; cannot extract usage_metadata.")
    first = messages[0]
    usage = getattr(first, "usage_metadata", None)
    if not isinstance(usage, dict):
        raise ValueError(
            f"AIMessage.usage_metadata missing or not a dict (got {type(usage).__name__}); "
            "extractor cannot compute actual cost."
        )
    return usage


def _token_count(usage: dict[str, Any], key: str) -> int:
    """Read a required non-negative token count from usage_metadata."""
    if key not in usage:
        raise ValueError(
            f"AIMessage.usage_metadata missing {key!r}; extractor cannot compute actual cost."
        )
    try:
        tokens = int(usage[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"AIMessage.usage_metadata[{key!r}] is not an integer token count; "
            "extractor cannot compute actual cost."
        ) from exc
    if tokens < 0:
        raise ValueError(
            f"AIMessage.usage_metadata[{key!r}] must be non-negative; "
            "extractor cannot compute actual cost."
        )
    return tokens


def _optional_detail_count(usage: dict[str, Any], key: str) -> int:
    details = usage.get("input_token_details")
    if details is None:
        return 0
    if not isinstance(details, dict):
        raise ValueError("AIMessage.usage_metadata['input_token_details'] must be a dict.")
    if key not in details:
        return 0
    return _token_count(details, key)


def _validate_rates(**rates: float | None) -> None:
    for name, rate in rates.items():
        if rate is not None and (not math.isfinite(rate) or rate < 0):
            raise ValueError(f"{name} must be a finite, non-negative price.")


def openai_cost(
    *,
    prompt_per_million_usd: float,
    completion_per_million_usd: float,
    cached_prompt_per_million_usd: float | None = None,
) -> CostFn:
    """Build a ``cost_fn`` for OpenAI-shaped responses.

    Reads ``input_tokens`` and ``output_tokens`` from
    ``AIMessage.usage_metadata`` (LangChain's normalized usage shape — works
    across the ``langchain-openai`` provider). Multiplies by the supplied
    per-million-token pricing and returns the total as a
    :class:`runcycles.Amount` in ``USD_MICROCENTS``.

    Pass ``cached_prompt_per_million_usd`` to price cache reads separately;
    without it, cached input uses the ordinary prompt rate for backward
    compatibility. The keyword-only API forces callers to label the rates so they
    can't accidentally swap input and output pricing (which differ ~4x
    on most OpenAI models)."""

    _validate_rates(
        prompt_per_million_usd=prompt_per_million_usd,
        completion_per_million_usd=completion_per_million_usd,
        cached_prompt_per_million_usd=cached_prompt_per_million_usd,
    )

    def _cost_fn(result: Any) -> Amount:
        usage = _extract_usage(result)
        input_tokens = _token_count(usage, "input_tokens")
        output_tokens = _token_count(usage, "output_tokens")
        cached_tokens = _optional_detail_count(usage, "cache_read")
        if cached_tokens > input_tokens:
            raise ValueError("cached input token count cannot exceed total input_tokens.")
        uncached_tokens = input_tokens - cached_tokens
        cached_rate = cached_prompt_per_million_usd
        if cached_rate is None:
            cached_rate = prompt_per_million_usd
        usd = (
            uncached_tokens * prompt_per_million_usd
            + cached_tokens * cached_rate
            + output_tokens * completion_per_million_usd
        ) / 1_000_000
        microcents = int(round(usd * _USD_TO_MICROCENTS))
        return Amount(unit=Unit.USD_MICROCENTS, amount=microcents)

    return _cost_fn


def anthropic_cost(
    *,
    input_per_million_usd: float,
    output_per_million_usd: float,
    cache_read_per_million_usd: float | None = None,
    cache_creation_per_million_usd: float | None = None,
    cache_creation_5m_per_million_usd: float | None = None,
    cache_creation_1h_per_million_usd: float | None = None,
) -> CostFn:
    """Build a ``cost_fn`` for Anthropic-shaped responses.

    Same underlying shape as :func:`openai_cost` — LangChain normalizes
    ``AIMessage.usage_metadata`` to ``input_tokens`` / ``output_tokens``
    across providers — so this factory exists for parameter-naming clarity
    rather than a different extraction path. Anthropic API docs label
    pricing as 'input' / 'output' tokens; OpenAI labels them
    'prompt' / 'completion'. Optional cache-read and cache-creation rates use
    normalized ``input_token_details``. The 5-minute and 1-hour arguments
    price Anthropic's cache-write tiers independently. A tier with no explicit
    rate uses ``cache_creation_per_million_usd`` and then the ordinary input
    rate as its fallback."""

    _validate_rates(
        input_per_million_usd=input_per_million_usd,
        output_per_million_usd=output_per_million_usd,
        cache_read_per_million_usd=cache_read_per_million_usd,
        cache_creation_per_million_usd=cache_creation_per_million_usd,
        cache_creation_5m_per_million_usd=cache_creation_5m_per_million_usd,
        cache_creation_1h_per_million_usd=cache_creation_1h_per_million_usd,
    )

    def _cost_fn(result: Any) -> Amount:
        usage = _extract_usage(result)
        input_tokens = _token_count(usage, "input_tokens")
        output_tokens = _token_count(usage, "output_tokens")
        cache_read_tokens = _optional_detail_count(usage, "cache_read")
        cache_creation_tokens = _optional_detail_count(usage, "cache_creation")
        cache_creation_5m_tokens = _optional_detail_count(usage, "ephemeral_5m_input_tokens")
        cache_creation_1h_tokens = _optional_detail_count(usage, "ephemeral_1h_input_tokens")
        tiered_creation_tokens = cache_creation_5m_tokens + cache_creation_1h_tokens
        if tiered_creation_tokens > cache_creation_tokens:
            raise ValueError("cache-creation tier counts cannot exceed cache_creation tokens.")
        standard_input_tokens = input_tokens - cache_read_tokens - cache_creation_tokens
        if standard_input_tokens < 0:
            raise ValueError("cache token counts cannot exceed total input_tokens.")
        cache_read_rate = cache_read_per_million_usd
        if cache_read_rate is None:
            cache_read_rate = input_per_million_usd
        cache_creation_rate = cache_creation_per_million_usd
        if cache_creation_rate is None:
            cache_creation_rate = input_per_million_usd
        cache_creation_5m_rate = cache_creation_5m_per_million_usd
        if cache_creation_5m_rate is None:
            cache_creation_5m_rate = cache_creation_rate
        cache_creation_1h_rate = cache_creation_1h_per_million_usd
        if cache_creation_1h_rate is None:
            cache_creation_1h_rate = cache_creation_rate
        unclassified_creation_tokens = cache_creation_tokens - tiered_creation_tokens
        usd = (
            standard_input_tokens * input_per_million_usd
            + cache_read_tokens * cache_read_rate
            + unclassified_creation_tokens * cache_creation_rate
            + cache_creation_5m_tokens * cache_creation_5m_rate
            + cache_creation_1h_tokens * cache_creation_1h_rate
            + output_tokens * output_per_million_usd
        ) / 1_000_000
        microcents = int(round(usd * _USD_TO_MICROCENTS))
        return Amount(unit=Unit.USD_MICROCENTS, amount=microcents)

    return _cost_fn


__all__ = ["anthropic_cost", "openai_cost"]
