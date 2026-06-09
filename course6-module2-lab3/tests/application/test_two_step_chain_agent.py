from unittest.mock import MagicMock

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.two_step_chain_agent import TwoStepChainAgent


def _make_tool(name: str, return_value: str):
    @tool
    def _t(arg: str) -> str:
        """Test tool."""
        return f"{name}({arg})={return_value}"

    _t.name = name  # type: ignore[misc]
    return _t


def _make_llm(*responses: AIMessage) -> MagicMock:
    bound = MagicMock()
    bound.invoke.side_effect = list(responses)
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound
    return llm


def test_handles_two_step_flow() -> None:
    """The agent's happy path — exactly two tool calls then a summary."""
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "step1", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "step2", "args": {"arg": "in"}, "id": "call_2", "type": "tool_call"}],
        ),
        AIMessage(content="summary"),
    )
    agent = TwoStepChainAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    assert agent.run("two-step") == "summary"