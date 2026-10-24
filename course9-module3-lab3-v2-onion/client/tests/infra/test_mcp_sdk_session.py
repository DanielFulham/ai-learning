import asyncio

import pytest
from mcp import StdioServerParameters
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
)

from infra.mcp_sdk_session import McpSdkSession
from interfaces.approval_source_interface import ApprovalAnswer, ApprovalSourceInterface


class _ScriptedUser:
    """An approval source that answers with a fixed choice and remembers what it was asked."""

    def __init__(self, answer: ApprovalAnswer = "accept") -> None:
        self.answer: ApprovalAnswer = answer
        self.questions: list[str] = []

    async def decide(self, message: str) -> ApprovalAnswer:
        self.questions.append(message)
        return self.answer


class _BrokenUser:
    async def decide(self, message: str) -> ApprovalAnswer:
        raise RuntimeError("TEST_BROKEN_SOURCE")


def _make_server() -> MCPServer:
    server = MCPServer("TEST_SERVER")

    @server.tool(description="TEST_ECHO_DESCRIPTION")
    def echo(text: str) -> str:
        return text

    @server.tool()
    def broken() -> str:
        raise ToolError("TEST_BROKEN")

    def add_asking_tool(tool_name: str, properties: dict[str, dict[str, str]]) -> None:
        async def asker(name: str, ctx: Context) -> str | InputRequiredResult:
            responses = ctx.input_responses
            if not responses:
                return InputRequiredResult(
                    input_requests={
                        "approval": ElicitRequest(
                            method="elicitation/create",
                            params=ElicitRequestFormParams(
                                message=f"Run risky on {name}?",
                                requested_schema={"type": "object", "properties": properties},
                            ),
                        )
                    },
                    request_state=f"TEST_{name}",
                )
            answer = responses["approval"]
            assert isinstance(answer, ElicitResult)
            return f"{name}: {answer.action} {answer.content}"

        server.tool(name=tool_name)(asker)

    add_asking_tool("risky", {"value": {"type": "boolean"}})
    add_asking_tool("risky_renamed", {"approve": {"type": "boolean"}})
    add_asking_tool("risky_no_flag", {"note": {"type": "string"}})

    @server.resource("file://TEST/one", name="TEST_RESOURCE", description="TEST_RESOURCE_DESC")
    def one() -> str:
        return "TEST_RESOURCE_TEXT"

    @server.prompt(description="TEST_PROMPT_DESC")
    def review(topic: str) -> str:
        return f"Review {topic}"

    return server


async def _connected() -> McpSdkSession:
    session = McpSdkSession(_make_server())
    await session.connect()
    return session


# --- connection ----------------------------------------------------------------------------


def test_given_a_session_that_is_not_connected_when_a_method_is_called_then_it_says_so() -> None:
    async def run() -> None:
        await McpSdkSession(_make_server()).list_tools()

    with pytest.raises(RuntimeError, match="Not connected"):
        asyncio.run(run())


def test_given_a_connected_session_when_connect_is_called_again_then_nothing_breaks() -> None:
    async def run() -> list[str]:
        session = await _connected()
        try:
            await session.connect()
            return [tool["name"] for tool in await session.list_tools()]
        finally:
            await session.close()

    assert sorted(asyncio.run(run())) == [
        "broken",
        "echo",
        "risky",
        "risky_no_flag",
        "risky_renamed",
    ]


def test_given_two_overlapping_connects_when_closed_once_then_no_connection_is_left_running() -> (
    None
):
    """Pins the lock: two Gradio events racing to connect must open one connection, not two.

    Without the lock a second connection would replace the first, which one close() could never
    stop, so a background task would still be running afterwards.
    """

    async def run() -> int:
        session = McpSdkSession(_make_server())
        await asyncio.gather(session.connect(), session.connect())
        await session.list_tools()
        await session.close()
        await asyncio.sleep(0)
        current = asyncio.current_task()
        return len([task for task in asyncio.all_tasks() if task is not current])

    assert asyncio.run(run()) == 0


def test_given_a_session_connected_in_one_task_when_closed_from_another_then_it_closes() -> None:
    """Pins the dedicated session task: Lab 46 could not close a session from another task."""

    async def run() -> None:
        session = McpSdkSession(_make_server())
        await asyncio.create_task(session.connect())
        await asyncio.create_task(session.close())
        await session.list_tools()

    with pytest.raises(RuntimeError, match="Not connected"):
        asyncio.run(run())


def test_given_a_session_closed_twice_when_the_second_close_runs_then_nothing_breaks() -> None:
    async def run() -> None:
        session = await _connected()
        await session.close()
        await session.close()

    asyncio.run(run())


def test_given_a_target_that_cannot_start_when_connected_then_it_raises_and_stays_closed() -> None:
    """Pins that a failed connect leaves a session that says "not connected", not a broken one."""

    async def run() -> None:
        session = McpSdkSession(StdioServerParameters(command="TEST_NO_SUCH_COMMAND_XYZ"))
        with pytest.raises(Exception):
            await session.connect()
        await session.list_tools()

    with pytest.raises(RuntimeError, match="Not connected"):
        asyncio.run(run())


def test_given_a_failed_connect_when_connect_is_called_again_then_it_tries_again() -> None:
    async def run() -> int:
        session = McpSdkSession(StdioServerParameters(command="TEST_NO_SUCH_COMMAND_XYZ"))
        failures = 0
        for _ in range(2):
            try:
                await session.connect()
            except Exception:
                failures += 1
        return failures

    assert asyncio.run(run()) == 2


def test_given_a_session_never_connected_when_closed_then_nothing_breaks() -> None:
    asyncio.run(McpSdkSession(_make_server()).close())


def test_given_a_closed_session_when_a_method_is_called_then_it_says_not_connected() -> None:
    async def run() -> None:
        session = await _connected()
        await session.close()
        await session.list_tools()

    with pytest.raises(RuntimeError, match="Not connected"):
        asyncio.run(run())


# --- tools ---------------------------------------------------------------------------------


def test_given_a_server_when_tools_are_listed_then_each_has_a_name_description_and_schema() -> None:
    async def run() -> dict[str, str]:
        session = await _connected()
        try:
            tools = await session.list_tools()
            assert all(
                "properties" in tool["input_schema"] for tool in tools if tool["name"] == "echo"
            )
            return {tool["name"]: tool["description"] for tool in tools}
        finally:
            await session.close()

    assert asyncio.run(run())["echo"] == "TEST_ECHO_DESCRIPTION"


def test_given_a_plain_tool_when_called_then_its_text_comes_back_without_an_error() -> None:
    async def run() -> tuple[list[str], bool]:
        session = await _connected()
        try:
            result = await session.call_tool("echo", {"text": "TEST_TEXT"}, _ScriptedUser())
            return result["texts"], result["is_error"]
        finally:
            await session.close()

    assert asyncio.run(run()) == (["TEST_TEXT"], False)


def test_given_a_tool_that_fails_when_called_then_the_result_is_flagged_as_an_error() -> None:
    async def run() -> tuple[str, bool]:
        session = await _connected()
        try:
            result = await session.call_tool("broken", {}, _ScriptedUser())
            return " ".join(result["texts"]), result["is_error"]
        finally:
            await session.close()

    text, is_error = asyncio.run(run())
    assert is_error is True
    assert "TEST_BROKEN" in text


def test_given_a_plain_tool_when_called_then_the_user_is_never_asked() -> None:
    user = _ScriptedUser()

    async def run() -> None:
        session = await _connected()
        try:
            await session.call_tool("echo", {"text": "TEST"}, user)
        finally:
            await session.close()

    asyncio.run(run())
    assert user.questions == []


# --- the server's question -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("accept", "thing: accept {'value': True}"),
        ("decline", "thing: decline None"),
        ("cancel", "thing: cancel None"),
    ],
)
def test_given_a_tool_that_asks_when_the_user_answers_then_the_server_gets_that_answer(
    answer: ApprovalAnswer, expected: str
) -> None:
    async def run() -> str:
        session = await _connected()
        try:
            result = await session.call_tool("risky", {"name": "thing"}, _ScriptedUser(answer))
            return result["texts"][0]
        finally:
            await session.close()

    assert asyncio.run(run()) == expected


def test_given_a_question_whose_flag_has_another_name_when_accepted_then_that_name_is_used() -> (
    None
):
    """Pins that the answer is built from the server's own schema, not a name written here."""

    async def run() -> str:
        session = await _connected()
        try:
            result = await session.call_tool("risky_renamed", {"name": "thing"}, _ScriptedUser())
            return result["texts"][0]
        finally:
            await session.close()

    assert asyncio.run(run()) == "thing: accept {'approve': True}"


def test_given_a_question_with_no_yes_or_no_field_when_asked_then_it_counts_as_cancel() -> None:
    """Pins fail-closed: if there is nowhere to put a yes, the answer is never an accept."""
    user = _ScriptedUser("accept")

    async def run() -> str:
        session = await _connected()
        try:
            result = await session.call_tool("risky_no_flag", {"name": "thing"}, user)
            return result["texts"][0]
        finally:
            await session.close()

    assert asyncio.run(run()) == "thing: cancel None"


def test_given_a_tool_that_asks_when_called_then_the_user_sees_the_servers_exact_question() -> None:
    """Pins F-L47-3: what the person is asked is the server's own text, untouched."""
    user = _ScriptedUser()

    async def run() -> None:
        session = await _connected()
        try:
            await session.call_tool("risky", {"name": "thing"}, user)
        finally:
            await session.close()

    asyncio.run(run())
    assert user.questions == ["Run risky on thing?"]


def test_given_an_approval_source_that_raises_when_the_server_asks_then_it_counts_as_cancel() -> (
    None
):
    """Pins fail-closed: a broken source must never turn into an accept."""

    async def run() -> str:
        session = await _connected()
        try:
            result = await session.call_tool("risky", {"name": "thing"}, _BrokenUser())
            return result["texts"][0]
        finally:
            await session.close()

    assert asyncio.run(run()) == "thing: cancel None"


def test_given_two_calls_at_once_when_each_has_its_own_source_then_each_is_asked_only_its_own() -> (
    None
):
    """Pins F-L47-18: one shared connection, but a question only reaches the call that caused it."""
    first, second = _ScriptedUser("accept"), _ScriptedUser("decline")

    async def run() -> tuple[str, str]:
        session = await _connected()
        try:
            one, two = await asyncio.gather(
                session.call_tool("risky", {"name": "ONE"}, first),
                session.call_tool("risky", {"name": "TWO"}, second),
            )
            return one["texts"][0], two["texts"][0]
        finally:
            await session.close()

    one_text, two_text = asyncio.run(run())
    assert first.questions == ["Run risky on ONE?"]
    assert second.questions == ["Run risky on TWO?"]
    assert one_text.startswith("ONE: accept")
    assert two_text.startswith("TWO: decline")


def test_given_a_scripted_user_when_typed_as_an_approval_source_then_it_fits_the_interface() -> (
    None
):
    """Pins that the test double satisfies the real interface, so the tests above mean something."""
    source: ApprovalSourceInterface = _ScriptedUser()
    assert source is not None


# --- resources and prompts -----------------------------------------------------------------


def test_given_a_server_when_resources_are_listed_then_each_has_a_uri_name_and_description() -> (
    None
):
    async def run() -> list[tuple[str, str, str]]:
        session = await _connected()
        try:
            return [(r["uri"], r["name"], r["description"]) for r in await session.list_resources()]
        finally:
            await session.close()

    assert asyncio.run(run()) == [("file://TEST/one", "TEST_RESOURCE", "TEST_RESOURCE_DESC")]


def test_given_a_resource_when_read_then_its_text_comes_back() -> None:
    async def run() -> list[str]:
        session = await _connected()
        try:
            return await session.read_resource("file://TEST/one", _ScriptedUser())
        finally:
            await session.close()

    assert asyncio.run(run()) == ["TEST_RESOURCE_TEXT"]


def test_given_a_server_when_prompts_are_listed_then_each_has_a_name_and_argument_names() -> None:
    async def run() -> list[tuple[str, list[str]]]:
        session = await _connected()
        try:
            return [(p["name"], p["argument_names"]) for p in await session.list_prompts()]
        finally:
            await session.close()

    assert asyncio.run(run()) == [("review", ["topic"])]


def test_given_a_prompt_when_fetched_with_arguments_then_the_rendered_messages_come_back() -> None:
    async def run() -> tuple[str, list[tuple[str, str]]]:
        session = await _connected()
        try:
            result = await session.get_prompt("review", {"topic": "TEST_TOPIC"})
            return result["description"], [(m["role"], m["text"]) for m in result["messages"]]
        finally:
            await session.close()

    description, messages = asyncio.run(run())
    assert description == "TEST_PROMPT_DESC"
    assert messages == [("user", "Review TEST_TOPIC")]
