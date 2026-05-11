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

from typing import Any

from runcycles import Amount, Unit

from langchain_runcycles._config import CostFn

# 1 USD = 100 cents = 10_000_000 micro-cents (10⁻⁸ USD per micro-cent).
_USD_TO_MICROCENTS = 100_000_000


def _extract_usage(result: Any) -> dict[str, int]:
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


def openai_cost(
    *,
    prompt_per_million_usd: float,
    completion_per_million_usd: float,
) -> CostFn:
    """Build a ``cost_fn`` for OpenAI-shaped responses.

    Reads ``input_tokens`` and ``output_tokens`` from
    ``AIMessage.usage_metadata`` (LangChain's normalized usage shape — works
    across the ``langchain-openai`` provider). Multiplies by the supplied
    per-million-token pricing and returns the total as a
    :class:`runcycles.Amount` in ``USD_MICROCENTS``.

    The keyword-only API forces callers to label the two rates so they
    can't accidentally swap input and output pricing (which differ ~4x
    on most OpenAI models)."""

    def _cost_fn(result: Any) -> Amount:
        usage = _extract_usage(result)
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        usd = (
            input_tokens * prompt_per_million_usd
            + output_tokens * completion_per_million_usd
        ) / 1_000_000
        microcents = int(round(usd * _USD_TO_MICROCENTS))
        return Amount(unit=Unit.USD_MICROCENTS, amount=microcents)

    return _cost_fn


def anthropic_cost(
    *,
    input_per_million_usd: float,
    output_per_million_usd: float,
) -> CostFn:
    """Build a ``cost_fn`` for Anthropic-shaped responses.

    Same underlying shape as :func:`openai_cost` — LangChain normalizes
    ``AIMessage.usage_metadata`` to ``input_tokens`` / ``output_tokens``
    across providers — so this factory exists for parameter-naming clarity
    rather than a different extraction path. Anthropic API docs label
    pricing as 'input' / 'output' tokens; OpenAI labels them
    'prompt' / 'completion'. The kwargs here match Anthropic's naming."""

    def _cost_fn(result: Any) -> Amount:
        usage = _extract_usage(result)
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        usd = (
            input_tokens * input_per_million_usd
            + output_tokens * output_per_million_usd
        ) / 1_000_000
        microcents = int(round(usd * _USD_TO_MICROCENTS))
        return Amount(unit=Unit.USD_MICROCENTS, amount=microcents)

    return _cost_fn


__all__ = ["anthropic_cost", "openai_cost"]
