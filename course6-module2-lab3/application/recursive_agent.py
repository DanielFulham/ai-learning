from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableLambda

from application.interfaces.video_agent_interface import VideoAgentInterface


class RecursiveAgent(VideoAgentInterface):
    """Agent that loops the tool-calling dispatch until the model is done.

    The model decides how many tool calls are needed. The loop terminates when
    the model emits an AIMessage with no tool_calls. This is the shape
    production agent frameworks (create_agent, LangGraph) use internally,
    with extra niceties around iteration caps, retries, and observability.

    No max-iteration cap in this implementation. Production code should add
    one — a misbehaving model can loop until the context length is exceeded.
    """

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]) -> None:
        self._tools = {t.name: t for t in tools}
        self._llm_with_tools = llm.bind_tools(tools)
        self._chain = self._build_chain()

    def run(self, query: str) -> str:
        result = self._chain.invoke({"query": query})
        final = result[-1].content
        if not isinstance(final, str):
            raise TypeError(f"Expected string content from final AIMessage, got {type(final).__name__}")
        return final

    def _execute_tool(self, tool_call: dict) -> ToolMessage:
        try:
            result = self._tools[tool_call["name"]].invoke(tool_call["args"])
            content = result if isinstance(result, str) else str(result)
        except Exception as e:
            content = f"Error: {str(e)}"
        return ToolMessage(content=content, tool_call_id=tool_call["id"])

    def _process_tool_calls(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        last = messages[-1]
        tool_messages = [
            self._execute_tool(tc) for tc in getattr(last, "tool_calls", [])
        ]
        updated = messages + tool_messages
        next_response = self._llm_with_tools.invoke(updated)
        return updated + [next_response]

    def _should_continue(self, messages: list[AnyMessage]) -> bool:
        return bool(getattr(messages[-1], "tool_calls", None))

    def _recurse(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        if self._should_continue(messages):
            return self._recurse(self._process_tool_calls(messages))
        return messages

    def _build_chain(self):
        def initial_messages(state: dict) -> list[AnyMessage]:
            return [HumanMessage(content=state["query"])]

        def first_invoke(messages: list[AnyMessage]) -> list[AnyMessage]:
            return messages + [self._llm_with_tools.invoke(messages)]

        return (
            RunnableLambda(initial_messages)
            | RunnableLambda(first_invoke)
            | RunnableLambda(self._recurse)
        )