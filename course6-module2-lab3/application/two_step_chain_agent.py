from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from application.interfaces.video_agent_interface import VideoAgentInterface


class TwoStepChainAgent(VideoAgentInterface):
    """Agent that hardcodes exactly two LLM-tool round-trips before summarising.

    Works for queries that need exactly two tools in sequence (e.g.
    extract_video_id then fetch_transcript). Fails for queries needing
    more or fewer steps. Included to demonstrate the cost of hardcoded
    orchestration vs the recursive shape.
    """

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]) -> None:
        self._tools = {t.name: t for t in tools}
        self._llm_with_tools = llm.bind_tools(tools)
        self._chain = self._build_chain()

    def run(self, query: str) -> str:
        result = self._chain.invoke({"query": query})
        if not isinstance(result, str):
            raise TypeError(f"Expected string from chain, got {type(result).__name__}")
        return result

    def _execute_tool(self, tool_call: dict) -> ToolMessage:
        try:
            result = self._tools[tool_call["name"]].invoke(tool_call["args"])
            content = result if isinstance(result, str) else str(result)
        except Exception as e:
            content = f"Error: {str(e)}"
        return ToolMessage(content=content, tool_call_id=tool_call["id"])

    def _build_chain(self):
        initial_setup = RunnablePassthrough.assign(
            messages=lambda x: [HumanMessage(content=x["query"])]
        )

        first_llm_call = RunnablePassthrough.assign(
            ai_response=lambda x: self._llm_with_tools.invoke(x["messages"])
        )

        first_tool_processing = RunnablePassthrough.assign(
            tool_messages=lambda x: [
                self._execute_tool(tc) for tc in x["ai_response"].tool_calls
            ]
        ).assign(
            messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
        )

        second_llm_call = RunnablePassthrough.assign(
            ai_response2=lambda x: self._llm_with_tools.invoke(x["messages"])
        )

        second_tool_processing = RunnablePassthrough.assign(
            tool_messages2=lambda x: [
                self._execute_tool(tc) for tc in x["ai_response2"].tool_calls
            ]
        ).assign(
            messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
        )

        final_summary = RunnablePassthrough.assign(
            summary=lambda x: self._llm_with_tools.invoke(x["messages"]).content
        ) | RunnableLambda(lambda x: x["summary"])

        return (
            initial_setup
            | first_llm_call
            | first_tool_processing
            | second_llm_call
            | second_tool_processing
            | final_summary
        )