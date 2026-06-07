"""Single-turn tool-calling agent: bind → invoke → dispatch → re-invoke."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool


class ToolCallingAgent:
    """Dispatches LLM-requested tool calls and summarises the result.

    Single-turn: handles one round of tool calls per query. For multi-turn
    iteration (tool result triggers another tool call), use langchain.agents.create_agent.
    """

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]) -> None:
        self._llm_with_tools = llm.bind_tools(tools)
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    def run(self, query: str) -> str:
        chat_history: list[BaseMessage] = [HumanMessage(content=query)]
        response = self._llm_with_tools.invoke(chat_history)

        # No-tool branch: model answered directly.
        if not response.tool_calls:
            return str(response.content)

        # OpenAI requires one ToolMessage per tool_call_id. Iterate every call.
        tool_messages = [
            ToolMessage(
                content=str(self._tool_map[call["name"]].invoke(call["args"])),
                tool_call_id=call["id"],
            )
            for call in response.tool_calls
        ]

        chat_history.extend([response, *tool_messages])
        final = self._llm_with_tools.invoke(chat_history)
        return str(final.content)