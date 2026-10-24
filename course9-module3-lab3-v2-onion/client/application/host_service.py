import json
from collections.abc import Sequence

from application.interfaces.approval_desk_interface import ApprovalDeskInterface
from application.mappers import (
    choice_from_reply,
    to_prompt_info,
    to_rendered_prompt,
    to_resource_info,
    to_tool_call_result,
    to_tool_info,
)
from domain.conversation import ChatReply, Message, ToolActivity, ToolCall
from domain.tool_info import ToolInfo
from interfaces.llm_chat_model_interface import LlmChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface

_NO_TEXT = "The model returned no text."

_NO_PARAMETERS: dict[str, object] = {"type": "object", "properties": {}}

_SYNTHETIC_TOOLS = (
    ToolInfo(
        "mcp_list_resources",
        "List all available resources from the MCP server",
        _NO_PARAMETERS,
    ),
    ToolInfo(
        "mcp_read_resource",
        "Read a specific resource by URI from the MCP server",
        {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": (
                        "The URI of the resource to read (for example, 'file://audit/log')"
                    ),
                }
            },
            "required": ["uri"],
        },
    ),
    ToolInfo(
        "mcp_list_prompts",
        "List all available prompt templates from the MCP server",
        _NO_PARAMETERS,
    ),
    ToolInfo(
        "mcp_get_prompt",
        "Get a rendered prompt template from the MCP server",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name of the prompt template"},
                "arguments": {
                    "type": "object",
                    "description": (
                        "Arguments for the prompt template, as an object "
                        '(for example, {"operation": "write_file", "risk_level": "MEDIUM"})'
                    ),
                },
            },
            "required": ["name"],
        },
    ),
)


class HostService:
    """The AI host: a chat with a model that can use the MCP server's tools, plus four helper
    tools that expose the server's resources and prompts to the model.

    One round of tool calls per message. The service keeps no conversation and no approval
    state: the caller passes the history and its own session's approval desk in, so one shared
    service can serve many page sessions without any of them seeing another's chat or answering
    another's question. When the server asks before running a tool, the question waits on that
    desk; the user's next message, if it is a bare yes or no, answers it.
    """

    def __init__(self, session: McpSessionInterface, chat_model: LlmChatModelInterface) -> None:
        self._session = session
        self._chat_model = chat_model

    async def get_available_tools(self) -> list[ToolInfo]:
        """The server's tools in the shape the model is offered, then the four helper tools."""
        await self._session.connect()
        tools: list[ToolInfo] = []
        for raw in await self._session.list_tools():
            tool = to_tool_info(raw)
            parameters: dict[str, object] = {"type": "object", "properties": {}}
            if "properties" in tool.input_schema:
                parameters["properties"] = tool.input_schema["properties"]
            if "required" in tool.input_schema and tool.input_schema["required"]:
                parameters["required"] = tool.input_schema["required"]
            tools.append(
                ToolInfo(tool.name, tool.description or f"Execute {tool.name}", parameters)
            )
        return tools + list(_SYNTHETIC_TOOLS)

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, object], desk: ApprovalDeskInterface
    ) -> str:
        """Run a tool call: one of the four helper tools, or a real tool on the server."""
        await self._session.connect()

        if tool_name == "mcp_list_resources":
            return await self._list_resources()

        if tool_name == "mcp_read_resource":
            return await self._read_resource(arguments, desk)

        if tool_name == "mcp_list_prompts":
            return await self._list_prompts()

        if tool_name == "mcp_get_prompt":
            return await self._get_prompt(arguments)

        try:
            result = to_tool_call_result(await self._session.call_tool(tool_name, arguments, desk))
            if result.is_error:
                detail = result.texts[0] if result.texts else "the tool reported an error"
                return f"Error executing tool: {detail}"
            return result.texts[0] if result.texts else ""
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    async def _list_resources(self) -> str:
        try:
            resources = [to_resource_info(raw) for raw in await self._session.list_resources()]
        except Exception as e:
            return f"Error listing resources: {str(e)}"
        result = "Available resources:\n"
        for resource in resources:
            result += f"- {resource.uri}"
            if resource.name:
                result += f" ({resource.name})"
            if resource.description:
                result += f": {resource.description}"
            result += "\n"
        return result

    async def _read_resource(
        self, arguments: dict[str, object], desk: ApprovalDeskInterface
    ) -> str:
        uri = arguments.get("uri")
        if not isinstance(uri, str) or not uri:
            return "Error: URI is required"
        try:
            texts = await self._session.read_resource(uri, desk)
            return texts[0] if texts else ""
        except Exception as e:
            return f"Error reading resource: {str(e)}"

    async def _list_prompts(self) -> str:
        try:
            prompts = [to_prompt_info(raw) for raw in await self._session.list_prompts()]
        except Exception as e:
            return f"Error listing prompts: {str(e)}"
        result = "Available prompts:\n"
        for prompt in prompts:
            result += f"- {prompt.name}"
            if prompt.description:
                result += f": {prompt.description}"
            if prompt.argument_names:
                result += f" (args: {', '.join(prompt.argument_names)})"
            result += "\n"
        return result

    async def _get_prompt(self, arguments: dict[str, object]) -> str:
        name = arguments.get("name")
        if not isinstance(name, str) or not name:
            return "Error: Prompt name is required"
        # The prompt's arguments belong in the `arguments` object. A model may also put them
        # beside `name`, so those are accepted when the object is empty.
        nested = arguments.get("arguments")
        if isinstance(nested, dict) and nested:
            raw_arguments: dict[object, object] = nested
        else:
            raw_arguments = {k: v for k, v in arguments.items() if k not in ("name", "arguments")}
        prompt_arguments: dict[str, str] = {}
        for key, value in raw_arguments.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return f"Error: prompt argument {key!r} must be text, not {value!r}"
            prompt_arguments[key] = value
        try:
            rendered = to_rendered_prompt(await self._session.get_prompt(name, prompt_arguments))
            result = f"Prompt: {name}\n\n"
            for message in rendered.messages:
                result += f"[{message.role}]: {message.text}\n\n"
            return result
        except Exception as e:
            return f"Error getting prompt: {str(e)}"

    async def _run_tool_call(self, call: ToolCall, desk: ApprovalDeskInterface) -> str:
        """Run one tool call the model asked for. Arguments the model got wrong are reported back
        to it as the tool's result, so it can correct itself, instead of ending the chat."""
        try:
            arguments = json.loads(call.arguments_json)
        except json.JSONDecodeError as e:
            return f"Error: the tool arguments were not valid JSON: {e}"
        if not isinstance(arguments, dict):
            return "Error: the tool arguments must be a JSON object"
        return await self.execute_tool(call.name, arguments, desk)

    async def chat(
        self, user_message: str, history: Sequence[Message], desk: ApprovalDeskInterface
    ) -> ChatReply:
        """Answer a message, running the tools the model asks for (one round).

        If a question is waiting on `desk`, the message is read as its answer instead: a bare
        yes or no answers it, anything else just repeats the question. Nothing reaches the model.
        """
        waiting = desk.pending_message()
        if waiting is not None:
            choice = choice_from_reply(user_message)
            if choice is None:
                text = f"A question is waiting for your answer:\n{waiting}\nReply yes or no."
                return ChatReply(text, history=tuple(history))
            if desk.respond(choice):
                return ChatReply(f"Sent your answer ({choice.value}).", history=tuple(history))

        await self._session.connect()
        messages = [*history, Message(role="user", text=user_message)]

        tools = await self.get_available_tools()
        reply = await self._chat_model.complete(tuple(messages), tools)
        if reply is None:
            return ChatReply("Error: No response from LLM", history=tuple(messages))

        if reply.tool_calls:
            messages.append(
                Message(role="assistant", text=reply.text or "", tool_calls=reply.tool_calls)
            )

            activity: list[ToolActivity] = []
            for call in reply.tool_calls:
                tool_result = str(await self._run_tool_call(call, desk))
                activity.append(ToolActivity(call.name, call.arguments_json, tool_result))
                messages.append(Message(role="tool", text=tool_result, tool_call_id=call.id))

            final = await self._chat_model.complete(tuple(messages), [])
            if final is None:
                message = "Error: No response from LLM after tool execution"
                return ChatReply(message, tuple(activity), tuple(messages))

            messages.append(Message(role="assistant", text=final.text))
            return ChatReply(final.text or _NO_TEXT, tuple(activity), tuple(messages))

        messages.append(Message(role="assistant", text=reply.text))
        return ChatReply(reply.text or _NO_TEXT, history=tuple(messages))
