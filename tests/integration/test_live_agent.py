"""Construction smoke tests against a real LangChain agent.

These tests don't make network calls — they verify the middleware classes can
actually be plugged into LangChain's `create_agent` without the runtime
rejecting the contract (wrong base class, missing required hooks, bad
@hook_config metadata, etc.).

If a future LangChain release breaks our middleware shape, these are the tests
that catch it before users do.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from runcycles import Action, CyclesClient, CyclesConfig, CyclesResponse, Subject

from langchain_runcycles import CyclesFanOutGate, CyclesModelGate, CyclesToolGate


def test_classes_satisfy_AgentMiddleware_protocol() -> None:
    """All three middleware classes must inherit from AgentMiddleware so
    create_agent accepts them in its `middleware=[...]` list."""
    assert issubclass(CyclesToolGate, AgentMiddleware)
    assert issubclass(CyclesFanOutGate, AgentMiddleware)
    assert issubclass(CyclesModelGate, AgentMiddleware)


def test_create_agent_accepts_tool_gate(sync_client: CyclesClient, subject: Any, action: Any) -> None:
    """A real `create_agent` call with CyclesToolGate as middleware succeeds."""

    @tool
    def noop(x: str) -> str:
        """A no-op tool."""
        return x

    fake_model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")

    agent = create_agent(model=fake_model, tools=[noop], middleware=[gate])
    assert agent is not None


def test_create_agent_accepts_fanout_gate() -> None:
    """A real `create_agent` call with CyclesFanOutGate as middleware succeeds.
    Verifies the @hook_config(can_jump_to=["end"]) annotation is well-formed."""

    @tool
    def noop(x: str) -> str:
        """A no-op tool."""
        return x

    fake_model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    fanout = CyclesFanOutGate(5)

    agent = create_agent(model=fake_model, tools=[noop], middleware=[fanout])
    assert agent is not None


def test_create_agent_accepts_both_middleware(
    sync_client: CyclesClient, subject: Any, action: Any
) -> None:
    """Both middleware composed in a single create_agent call — the shape we expect
    most users to land on (fan-out cap + per-tool authorization)."""

    @tool
    def noop(x: str) -> str:
        """A no-op tool."""
        return x

    fake_model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    fanout = CyclesFanOutGate(
        5,
        client=sync_client,
        subject=Subject(tenant="acme"),
        action=Action(kind="model.turn", name="research"),
    )
    tool_gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")

    agent = create_agent(model=fake_model, tools=[noop], middleware=[fanout, tool_gate])
    assert agent is not None


def test_create_agent_accepts_model_gate(sync_client: CyclesClient, subject: Any) -> None:
    """A real `create_agent` call with CyclesModelGate as middleware succeeds.
    Verifies the wrap_model_call hook annotation is well-formed."""

    @tool
    def noop(x: str) -> str:
        """A no-op tool."""
        return x

    fake_model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    model_gate = CyclesModelGate(
        sync_client,
        subject=subject,
        action=Action(kind="llm.completion", name="fake-model"),
        mode="decide",
    )

    agent = create_agent(model=fake_model, tools=[noop], middleware=[model_gate])
    assert agent is not None


def test_create_agent_accepts_full_triad(
    sync_client: CyclesClient, subject: Any, action: Any
) -> None:
    """All three middleware classes composed in a single create_agent call.

    This is the v0.1.5+ canonical shape: fan-out -> model -> tool ordering as
    the recommended composition pattern. Verifies the three classes don't
    interfere with each other's hook registration.
    """

    @tool
    def noop(x: str) -> str:
        """A no-op tool."""
        return x

    fake_model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    fanout = CyclesFanOutGate(
        5,
        client=sync_client,
        subject=Subject(tenant="acme"),
        action=Action(kind="model.turn", name="research"),
    )
    model_gate = CyclesModelGate(
        sync_client,
        subject=subject,
        action=Action(kind="llm.completion", name="fake-model"),
        mode="decide",
    )
    tool_gate = CyclesToolGate(sync_client, subject=subject, action=action, mode="decide")

    agent = create_agent(
        model=fake_model,
        tools=[noop],
        middleware=[fanout, model_gate, tool_gate],
    )
    assert agent is not None


@pytest.fixture
def subject() -> Subject:  # noqa: F811 - integration tests don't import the package conftest
    return Subject(tenant="acme", agent="bot")


@pytest.fixture
def action() -> Action:  # noqa: F811
    return Action(kind="tool.call", name="noop")


@pytest.fixture
def sync_client() -> CyclesClient:  # noqa: F811
    config = CyclesConfig(base_url="http://test", api_key="test-key")
    client = CyclesClient(config)
    client.decide = MagicMock(  # type: ignore[method-assign]
        return_value=CyclesResponse.success(200, {"decision": "ALLOW", "affected_scopes": []})
    )
    return client
