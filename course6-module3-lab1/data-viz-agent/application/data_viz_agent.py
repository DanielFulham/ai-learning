"""DataVizAgent: orchestrates an LLM and a Python REPL tool against a DataFrame.

Wraps langchain.agents.create_agent and exposes two public methods:

- run(query) returns just the final answer, suitable for production callers
  that only care about the agent's output
- run_with_trace(query) returns a structured AgentTrace including every tool
  call the agent made, suitable for demos, debugging, and observability

The agent doesn't know which LLM provider built the model, where charts get
saved, or what's in the DataFrame. All of that is configured in the container
and injected at construction time.

Only one orchestration shape is needed here: recursive tool calling until the
LLM stops emitting tool calls. create_agent provides this on top of LangGraph,
so we delegate to it rather than implementing the loop ourselves. If a future
variant needed manual instrumentation around each tool call (the way Lab 23's
ToolCallingAgent does), that would be a sibling class behind a DataVizAgentInterface.
For now there is no such variation, so the interface stays unwritten.
"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from domain.agent_trace import AgentTrace, ToolCallRecord


class DataVizAgent:
    """A data analysis agent backed by an LLM and a Python REPL tool."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        system_prompt: str,
    ) -> None:
        self._agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def run(self, query: str) -> str:
        """Run the agent against a natural-language query, return the final answer."""
        return self.run_with_trace(query).final_answer

    def run_with_trace(self, query: str) -> AgentTrace:
        """Run the agent and return the full trace of tool calls and final answer."""
        response = self._agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        messages = response["messages"]

        tool_calls = self._extract_tool_calls(messages)
        final_answer = self._extract_final_answer(messages)

        return AgentTrace(
            query=query,
            tool_calls=tool_calls,
            final_answer=final_answer,
        )

    @staticmethod
    def _extract_tool_calls(messages: list[BaseMessage]) -> tuple[ToolCallRecord, ...]:
        """Walk the message stream and pair each tool call with its result."""
        results_by_id: dict[str, str] = {
            msg.tool_call_id: str(msg.content)
            for msg in messages
            if isinstance(msg, ToolMessage)
        }

        records: list[ToolCallRecord] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    call_id = call.get("id") or ""
                    records.append(
                        ToolCallRecord(
                            name=call["name"],
                            args=dict(call["args"]),
                            result=results_by_id.get(call_id, ""),
                        )
                    )
        return tuple(records)

    @staticmethod
    def _extract_final_answer(messages: list) -> str:
        """Return the content of the final AIMessage in the trace."""
        if not messages:
            raise RuntimeError("Agent produced no messages")
        final_message = messages[-1]
        if not isinstance(final_message, AIMessage):
            raise RuntimeError(
                f"Expected final message to be AIMessage, got {type(final_message).__name__}"
            )
        content = final_message.content
        if not isinstance(content, str):
            raise RuntimeError(
                f"Expected final message content to be str, got {type(content).__name__}"
            )
        return content
