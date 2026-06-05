"""Tests for DataWizardAgent — the application-layer orchestrator.

DataWizardAgent is currently a thin facade over a ToolCallingAgentInterface.
These tests verify that delegation, and pin the layer's role in the architecture.
The test suite uses a mocked tool agent — no LLM, no LangChain, no network.
"""
from unittest.mock import MagicMock

from application.data_wizard_agent import DataWizardAgent
from interfaces.tool_calling_agent_interface import ToolCallingAgentInterface


def test_ask_delegates_to_underlying_agent():
    """The application's ask() should forward to the wrapped agent's ask()."""
    mock_agent = MagicMock(spec=ToolCallingAgentInterface)
    mock_agent.ask.return_value = "canned response"

    wizard = DataWizardAgent(agent=mock_agent)

    result = wizard.ask("what's in the data?")

    assert result == "canned response"
    mock_agent.ask.assert_called_once_with("what's in the data?")


def test_ask_returns_underlying_agent_response_verbatim():
    """No transformation, no wrapping — the inner response is returned as-is."""
    mock_agent = MagicMock(spec=ToolCallingAgentInterface)
    mock_agent.ask.return_value = "The classification dataset has 569 rows."

    wizard = DataWizardAgent(agent=mock_agent)

    result = wizard.ask("describe it")

    assert result == "The classification dataset has 569 rows."


def test_each_ask_call_invokes_underlying_agent():
    """Multiple queries delegate independently — no internal caching at this layer."""
    mock_agent = MagicMock(spec=ToolCallingAgentInterface)
    mock_agent.ask.side_effect = ["first answer", "second answer", "third answer"]

    wizard = DataWizardAgent(agent=mock_agent)

    first = wizard.ask("question 1")
    second = wizard.ask("question 2")
    third = wizard.ask("question 3")

    assert first == "first answer"
    assert second == "second answer"
    assert third == "third answer"
    assert mock_agent.ask.call_count == 3

def test_ask_with_empty_string_still_delegates():
    """No client-side validation — orchestration delegates to the agent regardless."""
    mock_agent = MagicMock(spec=ToolCallingAgentInterface)
    mock_agent.ask.return_value = "I need a question."

    wizard = DataWizardAgent(agent=mock_agent)

    result = wizard.ask("")

    assert result == "I need a question."
    mock_agent.ask.assert_called_once_with("")