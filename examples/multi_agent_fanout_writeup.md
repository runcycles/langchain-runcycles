# Pre-execution budget authority for multi-tenant agents

A pattern walkthrough — what to build, why it matters, and how the
[`multi_agent_fanout.py`](multi_agent_fanout.py) example demonstrates it.

## The problem nobody likes to think about

You ship an LLM agent. It works. Users love it. Then:

- A tenant fires off a prompt that loops the agent 200 turns before something stops it. The model bill is $40. The user got nothing useful — the loop was unproductive.
- A free-tier tenant runs the agent 800 times in a day before your aggregate-usage cron job flags them. By the time you respond, the model spend on free traffic is 5x your paid traffic.
- A staff member's prompt asks the agent to "send a summary to the team." Eight emails go out before anyone notices the model misinterpreted "team."

These aren't model failures. The model did exactly what it was asked. They're **runtime-authority failures** — the agent had the *capability* to keep spending, the *capability* to repeat a tool, the *capability* to send the email. No one was watching at the moment the action happened.

## What runtime authority actually looks like

The pattern: every *spend-creating* or *side-effecting* operation gets a synchronous yes/no from an external policy + budget service *before* it executes. Three kinds of operations, three kinds of authority:

1. **Per-turn fan-out**: cap the number of model invocations per run, with an optional remote-policy check. This is the cheapest possible gate: just count `AIMessage`s.
2. **Per-model-call spend**: reserve the call's worst-case cost *before* the LLM responds, commit at actual reported usage *after* the response, release on exception. Net effect: every model token your agent consumes flows through a budget the runtime can deny.
3. **Per-tool-call authorization**: `decide()` first (cheap policy check — is this tenant allowed to call this tool right now?), reserve the tool's cost if allowed, commit on success. Tools with side effects get the same treatment as the model call itself.

These three gates compose. Run them in cheapest-to-most-expensive order so the most expensive checks happen only when the cheaper ones pass:

```
fan-out cap  →  model authorization  →  human-in-the-loop  →  tool authorization
```

When the fan-out gate halts at turn 20, no model call happens, so no tool call happens, so no human is asked. When the tenant's budget is exhausted at the model gate, the agent never reaches HITL or the tool. When the tool gate denies, it happens after any required human approval but still before the side effect. **The early gates save the cost of everything downstream; the tool gate is the final machine authorization before action.**

## Why this matters more in multi-tenant deployments

If you serve one customer per process, you can stop the world when something looks wrong. If you serve a thousand tenants on one runtime, you can't — one tenant's runaway can't impact another's latency.

That means per-tenant accounting has to be **per-call**, not per-batch. Aggregating spend nightly is too late. Aggregating per-request is fine for billing but doesn't *stop* anything. You need policy + budget that fire *before each operation* and key everything on the tenant identity. That's the discipline this pattern enforces.

## The demo

[`multi_agent_fanout.py`](multi_agent_fanout.py) is a research-and-publish agent: a research-and-summarize loop that culminates in a `send_email` tool call. The agent has four tools (`search`, `summarize`, `spawn_subagent`, `send_email`) and runs against `claude-sonnet-4-6`. The full middleware stack is:

```python
middleware = [
    CyclesFanOutGate(max_turns=20, client=client, ...),
    CyclesModelGate(client, mode="decide+reserve", cost_fn=anthropic_cost(...), ...),
    HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
    CyclesToolGate(client, mode="decide+reserve", ...),
]
```

The subject — *who* is acting — is resolved per-call from agent state, so the same agent process serves many tenants and each tenant's budget is enforced independently. Each gate's decision is keyed on the tenant identity at the moment of the call.

### What you can watch happen

Run three tenants through the same agent:

- **Tenant A** (`acme`) has full budget. The research agent fans out, summarizes, drafts an email, and pauses for human approval. Human approves. `CyclesToolGate` authorizes and commits the `send_email` reservation. Done.
- **Tenant B** (`globex`) has exhausted their per-day research budget. `CyclesModelGate.decide()` denies the first LLM call. The agent never reaches the research tools. Total spend on Cycles' side: one `decide()` API call. Total spend on Anthropic's side: zero.
- **Tenant C** (`hooli`) has budget but no allowance for `send_email` specifically. Research runs normally. The human can approve the proposed email, but `CyclesToolGate.decide()` still denies before the tool executes; the agent's reasoning loop sees the denial as a tool result and chooses an alternative path (printing the draft instead of sending). No email is sent.

The same `create_agent(...)` call serves all three. The runtime authority is in the middleware list, not branched into your application code.

### What the `cost_fn` extractor adds in v0.2.0

Before v0.2.0, `CyclesModelGate` reserved the worst-case `estimate` and committed at the same amount on success. Accurate enough for "is this tenant over budget?" — wrong for "what did we actually charge them?"

`cost_fn` closes that gap. The reservation still uses `estimate` as headroom (denials happen before the model runs, so it has to be a worst-case number), but the commit at success uses the model's actual reported token counts via:

```python
cost_fn=anthropic_cost(
    input_per_million_usd=3.00,
    output_per_million_usd=15.00,
)
```

Effect: your per-tenant budget ledger matches the Anthropic invoice line-by-line. A 50-token completion debits a 50-token cost, not a 2,500,000-microcent worst-case bucket.

If the extractor itself fails (provider response shape changes, missing `usage_metadata`, pricing miscalculation), the gate logs and falls back to `estimate`. A costing bug downgrades accuracy; it never erases a successful model result.

## When to reach for this versus simpler alternatives

This pattern is overkill if you're shipping one assistant to one user. Don't bring in the runtime-authority layer just because you can.

Reach for it when:

- **You have more than one tenant** sharing the same agent process and budgets need to be enforced per-tenant in real-time.
- **You're paying for the model** and a runaway loop costs you real money.
- **You have side-effecting tools** (email, payments, API writes) and a wrong call has a blast radius.
- **You need an external audit trail** of what the agent attempted, what was allowed, what was denied. Cycles writes every decision to a server you control.

For the single-tenant assistant case, [`tenant_budget_agent.py`](tenant_budget_agent.py) shows the minimal shape — same middleware, simpler subject/action resolution.

## Where to take this next

The example stops at the agent boundary. Realistic deployments extend it three ways:

1. **Long-term memory**: store per-tenant agent state in a LangGraph checkpointer, so denials and approvals carry across sessions. This is where the LangGraph state-management story compounds with the runtime-authority story.
2. **Observability**: every `decide()` and reservation is correlated by `X-Cycles-Trace-Id`. Connect that to your tracing backend (Datadog, Honeycomb, LangSmith) so the policy decision and the LangChain run trace are joinable.
3. **Tool-side `cost_fn`**: parity with the model gate is on the roadmap — once it ships, `CyclesToolGate` will commit at the tool's actual reported cost (e.g., an API call's billable response size) instead of the configured `estimate`.

The example is a runnable starting point. The pattern — pre-execution budget authority composed from cheap-to-expensive gates — is what holds up.
