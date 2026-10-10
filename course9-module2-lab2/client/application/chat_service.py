from application.mappers import to_tool_spec
from domain.conversation import Message, ToolCall, ToolResult
from interfaces.chat_model_interface import ChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface


class ChatService:
    """Answers a query with the model, running the server's tools when it asks."""

    def __init__(
        self,
        chat_model: ChatModelInterface,
        session: McpSessionInterface,
        user_input: UserInputInterface,
    ) -> None:
        self._chat_model = chat_model
        self._session = session
        self._user_input = user_input

    async def process_query(self, query: str) -> str:
        """Ask the model, run any tools it requests, and repeat until it gives a final answer."""
        tools = [to_tool_spec(raw) for raw in await self._session.list_tools()]
        messages = [Message(role="user", text=query)]

        reply = await self._chat_model.complete(messages, tools)
        while reply.tool_calls:
            messages.append(Message(role="assistant", text=reply.text, tool_calls=reply.tool_calls))
            results = [await self._run_tool(call) for call in reply.tool_calls]
            messages.append(Message(role="user", tool_results=tuple(results)))
            reply = await self._chat_model.complete(messages, tools)

        if reply.truncated:
            self._user_input.show(
                "Warning: response was cut off at max_tokens; a tool call in it was not run"
            )
        return reply.text

    async def _run_tool(self, call: ToolCall) -> ToolResult:
        try:
            result = await self._session.call_tool(call.name, call.arguments)
        except Exception as e:
            self._user_input.show(f"Error calling tool {call.name}: {e}")
            return ToolResult(call.id, f"Error: {e}", is_error=True)
        return ToolResult(call.id, result["text"], is_error=result["is_error"])
