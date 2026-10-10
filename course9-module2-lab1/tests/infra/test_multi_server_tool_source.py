import asyncio
from unittest.mock import AsyncMock, patch

from infra.multi_server_tool_source import MultiServerToolSource


def test_client_is_built_with_tool_name_prefix_on() -> None:
    with patch("infra.multi_server_tool_source.MultiServerMCPClient") as client_class:
        MultiServerToolSource("TEST_COMMAND", ["TEST_ARG"], "TEST_URL")
        assert client_class.call_args.kwargs == {"tool_name_prefix": True}


def test_client_is_given_one_stdio_and_one_http_server() -> None:
    with patch("infra.multi_server_tool_source.MultiServerMCPClient") as client_class:
        MultiServerToolSource("TEST_COMMAND", ["TEST_ARG"], "TEST_URL")
        connections = client_class.call_args.args[0]
        assert connections == {
            "stdio-client": {
                "command": "TEST_COMMAND",
                "args": ["TEST_ARG"],
                "transport": "stdio",
            },
            "http-client": {"url": "TEST_URL", "transport": "streamable_http"},
        }


def test_open_yields_the_tools_the_client_returns() -> None:
    with patch("infra.multi_server_tool_source.MultiServerMCPClient") as client_class:
        client_class.return_value.get_tools = AsyncMock(return_value=["TEST_TOOL"])
        source = MultiServerToolSource("TEST_COMMAND", [], "TEST_URL")

        async def run() -> list[object]:
            async with source.open() as tools:
                return list(tools)

        assert asyncio.run(run()) == ["TEST_TOOL"]
