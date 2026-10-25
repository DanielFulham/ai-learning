import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.elicitation import ElicitResult
from mcp.types import CallToolResult, TextContent

import config
from application.container import initialise_server
from interfaces.file_store_interface import FileStoreInterface


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


class _Run:
    """A real server wired by the container over temp folders, with a scripted user."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "root"
        self.state = tmp_path / "state"
        self.questions: list[str] = []

    def server(self) -> FastMCP:
        return initialise_server(root_dir=self.root, state_dir=self.state, name="TEST_SERVER")

    def call(self, tool: str, args: dict[str, object]) -> CallToolResult:
        async def handler(
            message: str, response_type: object, params: object, context: object
        ) -> ElicitResult:
            self.questions.append(message)
            return ElicitResult(action="accept", content={"value": True})

        async def run() -> CallToolResult:
            client = Client(self.server(), mode="auto", elicitation_handler=handler)
            async with client:
                return await client.call_tool_mcp(tool, args)

        return asyncio.run(run())


def test_given_the_container_when_it_builds_a_server_then_it_has_the_requested_name(
    tmp_path: Path,
) -> None:
    server = _Run(tmp_path).server()
    assert isinstance(server, FastMCP)
    assert server.name == "TEST_SERVER"


def test_given_two_calls_when_the_container_builds_servers_then_each_is_a_new_instance(
    tmp_path: Path,
) -> None:
    """Pins statelessness: the entry point decides what to keep, as the onion doc says."""
    run = _Run(tmp_path)
    assert run.server() is not run.server()


def test_given_the_default_config_when_the_folders_are_compared_then_neither_holds_the_other() -> (
    None
):
    """Pins F-L47-8: the policy file and audit log sit outside the folder the file tools reach."""
    assert not config.STATE_DIR.is_relative_to(config.ROOT_DIR)
    assert not config.ROOT_DIR.is_relative_to(config.STATE_DIR)


def test_given_a_state_folder_inside_the_root_when_the_server_is_built_then_it_is_refused(
    tmp_path: Path,
) -> None:
    """Pins F-L47-8 at build time: a wrong path must not put the policy where tools can reach."""
    with pytest.raises(ValueError, match="state"):
        initialise_server(root_dir=tmp_path, state_dir=tmp_path / "state")


def test_given_the_same_folder_for_root_and_state_when_the_server_is_built_then_it_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="state"):
        initialise_server(root_dir=tmp_path, state_dir=tmp_path)


def test_given_a_root_folder_inside_the_state_folder_when_the_server_is_built_then_it_is_allowed(
    tmp_path: Path,
) -> None:
    assert isinstance(initialise_server(root_dir=tmp_path / "files", state_dir=tmp_path), FastMCP)


def test_given_a_state_folder_inside_the_root_when_a_file_store_is_injected_then_no_check_is_made(
    tmp_path: Path,
) -> None:
    """The guard protects the default file store only; an injected store owns its own reach."""
    store = MagicMock(spec=FileStoreInterface)
    server = initialise_server(root_dir=tmp_path, state_dir=tmp_path / "state", file_store=store)
    assert isinstance(server, FastMCP)


def test_given_an_ask_write_when_approved_then_the_file_lands_in_the_root_folder(
    tmp_path: Path,
) -> None:
    run = _Run(tmp_path)
    result = run.call("write_file", {"filepath": "TEST.txt", "content": "TEST_BODY"})
    assert not result.is_error
    assert (run.root / "TEST.txt").read_text(encoding="utf-8") == "TEST_BODY"
    assert len(run.questions) == 1


def test_given_a_call_when_it_finishes_then_the_audit_log_is_in_the_state_folder(
    tmp_path: Path,
) -> None:
    run = _Run(tmp_path)
    run.call("write_file", {"filepath": "TEST.txt", "content": "TEST_BODY"})
    lines = (run.state / "audit.log").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["approval_requested", "approved", "succeeded"]
    assert not (run.root / "audit.log").exists()


def test_given_a_policy_file_in_the_state_folder_when_a_tool_is_called_then_it_applies(
    tmp_path: Path,
) -> None:
    run = _Run(tmp_path)
    run.state.mkdir()
    (run.state / "permissions.json").write_text('{"write_file": "allow"}', encoding="utf-8")
    result = run.call("write_file", {"filepath": "TEST.txt", "content": "TEST_BODY"})
    assert not result.is_error
    assert run.questions == []


def test_given_an_existing_audit_log_when_a_tool_reads_it_by_path_then_it_is_refused(
    tmp_path: Path,
) -> None:
    """Pins F-L47-8 end to end: read_file cannot reach the audit log by any relative path."""
    run = _Run(tmp_path)
    run.call("write_file", {"filepath": "TEST.txt", "content": "TEST_BODY"})
    outside = run.call("read_file", {"filepath": "../state/audit.log"})
    inside = run.call("read_file", {"filepath": "audit.log"})
    assert outside.is_error
    assert "outside the server root" in _text(outside)
    assert inside.is_error


def test_given_an_existing_policy_file_when_write_file_targets_it_then_the_policy_is_unchanged(
    tmp_path: Path,
) -> None:
    """Pins F-L47-8: the lab let an approved write replace the policy and wipe the audit log."""
    run = _Run(tmp_path)
    run.state.mkdir()
    policy = run.state / "permissions.json"
    policy.write_text('{"delete_file": "deny"}', encoding="utf-8")
    result = run.call("write_file", {"filepath": "../state/permissions.json", "content": "{}"})
    assert result.is_error
    assert policy.read_text(encoding="utf-8") == '{"delete_file": "deny"}'


def test_given_an_injected_file_store_when_a_tool_runs_then_the_container_uses_it(
    tmp_path: Path,
) -> None:
    store = MagicMock(spec=FileStoreInterface)
    store.read_text.return_value = "TEST_FROM_INJECTED_STORE"
    run = _Run(tmp_path)

    async def call() -> CallToolResult:
        server = initialise_server(state_dir=run.state, file_store=store, name="TEST_SERVER")
        async with Client(server, mode="auto") as client:
            return await client.call_tool_mcp("read_file", {"filepath": "TEST.txt"})

    assert _text(asyncio.run(call())) == "TEST_FROM_INJECTED_STORE"
