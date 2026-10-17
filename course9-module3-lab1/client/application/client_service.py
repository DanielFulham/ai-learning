import json

from application.formatters import (
    COMMANDS_LINE,
    CONNECTED,
    HELP_TEXT,
    WELCOME,
    format_history,
    format_prompt,
    format_rendered_prompt,
    format_resource_template,
    format_tool,
)
from application.mappers import (
    to_prompt_info,
    to_rendered_prompt,
    to_resource_template_info,
    to_tool_call_result,
    to_tool_info,
)
from domain.history_entry import HistoryEntry
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface


class ClientService:
    """The command-line client: tools, resource templates and prompts of one MCP server."""

    def __init__(self, session: McpSessionInterface, user_input: UserInputInterface) -> None:
        self._session = session
        self._user_input = user_input
        self._history: list[HistoryEntry] = []

    async def run(self) -> None:
        try:
            await self._session.connect()
            self._user_input.show(CONNECTED)
            await self._command_loop()
        except KeyboardInterrupt:
            self._user_input.show("\nExiting...")
        finally:
            await self._session.close()

    async def _command_loop(self) -> None:
        self._user_input.show(WELCOME)
        self._user_input.show(COMMANDS_LINE)

        while True:
            command = self._user_input.ask("> ").lower()
            if command == "quit":
                return
            await self._dispatch(command)

    async def _dispatch(self, command: str) -> None:
        try:
            if command == "tools":
                await self._tools()
            elif command == "call":
                await self._call()
            elif command == "resources":
                await self._resources()
            elif command == "read":
                await self._read()
            elif command == "prompts":
                await self._prompts()
            elif command == "prompt":
                await self._prompt()
            elif command == "history":
                self._user_input.show(format_history(self._history))
            elif command == "help":
                self._user_input.show(HELP_TEXT)
            else:
                self._user_input.show("  Unknown command")
        except json.JSONDecodeError:
            self._user_input.show("  Error: Invalid JSON format")
            self._user_input.show('  Hint: Use double quotes, for example, {"text": "hello"}')
        except Exception as e:
            self._user_input.show(f"  Error: {e}")
            if "not found" in str(e).lower():
                self._user_input.show("  Hint: Check the resource URI or filename")

    async def _tools(self) -> None:
        for raw in await self._session.list_tools():
            self._user_input.show(format_tool(to_tool_info(raw)))

    async def _call(self) -> None:
        tool_name = self._user_input.ask("  Tool name: ")
        self._user_input.show('  Arguments (as JSON, for example, {"text": "hello"}): ')
        arguments = json.loads(self._user_input.ask("  "))
        result = to_tool_call_result(await self._session.call_tool(tool_name, arguments))
        self._history.append(HistoryEntry(operation="tool", name=tool_name))
        for text in result.texts:
            self._user_input.show(f"  Result: {text}")

    async def _resources(self) -> None:
        templates = await self._session.list_resource_templates()
        if not templates:
            self._user_input.show("  No resources available")
            return
        for raw in templates:
            self._user_input.show(format_resource_template(to_resource_template_info(raw)))

    async def _read(self) -> None:
        uri = self._user_input.ask("  URI: ")
        for text in await self._session.read_resource(uri):
            self._user_input.show(f"\n{text}")

    async def _prompts(self) -> None:
        for raw in await self._session.list_prompts():
            self._user_input.show(format_prompt(to_prompt_info(raw)))

    async def _prompt(self) -> None:
        prompt_name = self._user_input.ask("  Prompt name: ")
        self._user_input.show("  Arguments (as JSON): ")
        arguments = json.loads(self._user_input.ask("  "))
        rendered = to_rendered_prompt(await self._session.get_prompt(prompt_name, arguments))
        self._user_input.show(format_rendered_prompt(rendered))
