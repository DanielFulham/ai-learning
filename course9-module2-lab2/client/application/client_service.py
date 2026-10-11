from urllib.parse import quote

from application.chat_service import ChatService
from application.formatters import format_directory_listing
from application.mappers import parse_directory_listing, parse_file_content, to_prompt_spec
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface

_MENU = """
Select from the Menu
1. Generate Documentation
2. Review Code
3. Read File
4. Read Current Directory
5. Converse with Agent
q. Quit
> """


class ClientService:
    """The menu-driven client: prompts, file resources and a conversation with the model."""

    def __init__(
        self,
        chat_service: ChatService,
        session: McpSessionInterface,
        user_input: UserInputInterface,
    ) -> None:
        self._chat = chat_service
        self._session = session
        self._user_input = user_input

    async def run(self) -> None:
        try:
            await self._session.connect()
            await self._menu()
        except Exception as e:
            self._user_input.show(f"Error: {type(e).__name__}: {e}")
        finally:
            await self._session.close()

    async def _menu(self) -> None:
        self._user_input.show("\nMCP Client Started!")
        self._user_input.show("Select from the menu or 'quit'/'q' to exit.")

        while True:
            choice = self._user_input.ask(_MENU).lower()
            if choice in ("q", "quit"):
                self._user_input.show("Exiting client...")
                return
            if choice == "1":
                await self._run_prompt("documentation_generator", use_tools=True)
            elif choice == "2":
                await self._run_prompt("code_review", use_tools=False)
            elif choice == "3":
                await self._read_file()
            elif choice == "4":
                await self._read_dir()
            elif choice == "5":
                await self._converse()
            else:
                self._user_input.show("Invalid choice. Please try again.")

    async def _converse(self) -> None:
        self._user_input.show("\nEntering conversation mode. Type 'quit' or 'q' to exit.")
        while True:
            query = self._user_input.ask("\nQuery: ")
            if query.lower() in ("quit", "q"):
                return
            if not query:
                self._user_input.show("Please enter a query")
                continue
            try:
                self._user_input.show("\n" + await self._chat.process_query(query))
            except Exception as e:
                self._user_input.show(f"Error processing query: {e}")

    async def _run_prompt(self, prompt_name: str, use_tools: bool) -> None:
        try:
            prompts = [to_prompt_spec(raw) for raw in await self._session.list_prompts()]
            spec = next((p for p in prompts if p.name == prompt_name), None)
            if spec is None:
                self._user_input.show(f"Prompt '{prompt_name}' not found")
                return

            arguments: dict[str, str] = {}
            for arg in spec.arguments:
                label = "required" if arg.required else "optional"
                answer = self._user_input.ask(f"{arg.name} ({label}): ")
                if not answer and arg.required:
                    self._user_input.show(f"Error: {arg.name} is required")
                    return
                if answer:
                    arguments[arg.name] = answer

            prompt_text = await self._session.get_prompt_text(prompt_name, arguments)
            self._user_input.show(await self._chat.process_query(prompt_text, use_tools=use_tools))
        except Exception as e:
            self._user_input.show(f"Error: {type(e).__name__}: {e}\n")

    async def _read_file(self) -> None:
        try:
            file_name = self._user_input.ask("Enter file path: ")
            text = await self._session.read_resource_text(f"file:///{quote(file_name, safe='')}")
            self._user_input.show(f"File Content:\n {parse_file_content(text)}")
        except Exception as e:
            self._user_input.show(f"Error reading file: {e}")

    async def _read_dir(self) -> None:
        try:
            text = await self._session.read_resource_text("dir://.")
            self._user_input.show(format_directory_listing(parse_directory_listing(text)))
        except Exception as e:
            self._user_input.show(f"Error reading directory: {e}")
