import asyncio
import json
from typing import Literal
from unittest.mock import MagicMock

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from mcp.types import CallToolResult, TextContent, TextResourceContents

from application.lab_server import build_server
from application.permission_gate import PermissionGate
from application.policy_book import PolicyBook
from interfaces.audit_log_interface import AuditLogInterface
from interfaces.file_store_interface import FileStoreInterface, PathOutsideRootError
from interfaces.policy_store_interface import InvalidPolicyError, PolicyStoreInterface

TEST_NAME = "TEST_SERVER"
Answer = Literal["accept", "decline", "cancel"]


class _Fixture:
    def __init__(self, rules: dict[str, str] | None = None) -> None:
        self.store = MagicMock(spec=FileStoreInterface)
        self.store.read_text.return_value = "TEST_CONTENT"
        self.audit = MagicMock(spec=AuditLogInterface)
        self.audit.read_text.return_value = ""
        self.policy_store = MagicMock(spec=PolicyStoreInterface)
        self.policy_store.load_rules.return_value = rules or {}
        self.questions: list[str] = []

    def client(self, answer: Answer | None = "accept", mode: str = "auto") -> Client:
        book = PolicyBook(self.policy_store)
        server = build_server(
            self.store, PermissionGate(book, self.audit), self.audit, book, TEST_NAME
        )
        if answer is None:
            return Client(server, mode=mode)

        async def handler(
            message: str, response_type: object, params: object, context: object
        ) -> ElicitResult:
            self.questions.append(message)
            content = {"value": True} if answer == "accept" else None
            return ElicitResult(action=answer, content=content)

        return Client(server, mode=mode, elicitation_handler=handler)

    def call(
        self,
        tool: str,
        args: dict[str, object],
        answer: Answer | None = "accept",
        mode: str = "auto",
    ) -> CallToolResult:
        async def run() -> CallToolResult:
            async with self.client(answer, mode) as client:
                return await client.call_tool_mcp(tool, args)

        return asyncio.run(run())


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


# --- the four tools ------------------------------------------------------------------------


def test_given_the_server_when_tools_are_listed_then_the_four_lab_tools_exist() -> None:
    async def run() -> set[str]:
        async with _Fixture().client() as client:
            return {tool.name for tool in await client.list_tools()}

    assert asyncio.run(run()) == {"read_file", "write_file", "delete_file", "execute_command"}


@pytest.mark.parametrize(
    ("tool", "read_only", "destructive"),
    [
        ("read_file", True, False),
        ("write_file", False, True),
        ("delete_file", False, True),
        ("execute_command", False, True),
    ],
)
def test_given_each_tool_when_listed_then_its_annotations_describe_its_behaviour(
    tool: str, read_only: bool, destructive: bool
) -> None:
    """Pins that the protocol's own hints replace the lab's text-only "(Risk: ...)" docstrings."""

    async def run() -> tuple[bool | None, bool | None]:
        async with _Fixture().client() as client:
            listed = {t.name: t for t in await client.list_tools()}
            annotations = listed[tool].annotations
            assert annotations is not None
            return annotations.read_only_hint, annotations.destructive_hint

    assert asyncio.run(run()) == (read_only, destructive)


@pytest.mark.parametrize(
    ("tool", "risk"),
    [
        ("read_file", "LOW"),
        ("write_file", "MEDIUM"),
        ("delete_file", "HIGH"),
        ("execute_command", "CRITICAL"),
    ],
)
def test_given_each_tool_when_listed_then_its_description_states_the_risk_level(
    tool: str, risk: str
) -> None:
    async def run() -> str:
        async with _Fixture().client() as client:
            listed = {t.name: t for t in await client.list_tools()}
            return listed[tool].description or ""

    assert f"Risk: {risk}" in asyncio.run(run())


def test_given_an_allowed_read_when_called_then_the_file_text_comes_back_unasked() -> None:
    fixture = _Fixture()
    result = fixture.call("read_file", {"filepath": "TEST.txt"})
    assert not result.is_error
    assert _text(result) == "TEST_CONTENT"
    fixture.store.read_text.assert_called_once_with("TEST.txt")
    assert fixture.questions == []


def test_given_a_read_outside_the_root_when_called_then_it_is_an_error_result() -> None:
    """Pins F-L47-6: the refusal comes from the store and is reported as a tool error."""
    fixture = _Fixture()
    fixture.store.read_text.side_effect = PathOutsideRootError("TEST_OUTSIDE")
    result = fixture.call("read_file", {"filepath": "../TEST.txt"})
    assert result.is_error
    assert "TEST_OUTSIDE" in _text(result)


def test_given_a_read_of_a_missing_file_when_called_then_it_is_an_error_result() -> None:
    fixture = _Fixture()
    fixture.store.read_text.side_effect = FileNotFoundError("TEST_MISSING")
    result = fixture.call("read_file", {"filepath": "TEST.txt"})
    assert result.is_error
    assert "TEST_MISSING" in _text(result)


def test_given_an_ask_write_and_an_accept_when_called_then_the_file_is_written() -> None:
    fixture = _Fixture()
    result = fixture.call("write_file", {"filepath": "TEST.txt", "content": "TEST_BODY"})
    assert not result.is_error
    assert _text(result) == "Successfully wrote to TEST.txt"
    fixture.store.write_text.assert_called_once_with("TEST.txt", "TEST_BODY")


def test_given_an_ask_write_when_called_then_the_user_is_asked_once_with_the_arguments() -> None:
    """Pins F-L47-3: the question the user answers carries the tool and what it will do."""
    fixture = _Fixture()
    fixture.call("write_file", {"filepath": "TEST.txt", "content": "TEST_BODY"})
    assert len(fixture.questions) == 1
    assert "write_file" in fixture.questions[0]
    assert "TEST.txt" in fixture.questions[0]
    assert "TEST_BODY" in fixture.questions[0]


@pytest.mark.parametrize("answer", ["decline", "cancel"])
def test_given_an_ask_write_and_no_approval_when_called_then_nothing_is_written(
    answer: Answer,
) -> None:
    fixture = _Fixture()
    result = fixture.call("write_file", {"filepath": "TEST.txt", "content": "x"}, answer)
    assert result.is_error
    fixture.store.write_text.assert_not_called()


def test_given_a_client_with_no_handler_when_an_ask_tool_is_called_then_nothing_is_written() -> (
    None
):
    """Pins fail-closed: a client that cannot answer never gets the write done."""
    fixture = _Fixture()
    try:
        result = fixture.call("write_file", {"filepath": "TEST.txt", "content": "x"}, None)
    except Exception:  # a client that cannot answer may raise instead of returning an error
        result = None
    assert result is None or result.is_error
    fixture.store.write_text.assert_not_called()


def test_given_a_legacy_client_when_an_ask_tool_is_called_then_nothing_is_written() -> None:
    """Pins a library fact: the input-required result only exists at 2026-07-28.

    A client on the old handshake cannot be asked, so the call fails and the write never runs.
    """
    fixture = _Fixture()
    with pytest.raises(Exception, match="2026-07-28"):
        fixture.call("write_file", {"filepath": "TEST.txt", "content": "x"}, "accept", "legacy")
    fixture.store.write_text.assert_not_called()


def test_given_the_default_deny_when_a_delete_is_called_then_it_is_refused_unasked() -> None:
    fixture = _Fixture()
    result = fixture.call("delete_file", {"filepath": "TEST.txt"})
    assert result.is_error
    assert _text(result) == "Permission denied for tool: delete_file"
    fixture.store.delete.assert_not_called()
    assert fixture.questions == []


def test_given_an_ask_delete_and_an_accept_when_called_then_the_file_is_deleted() -> None:
    fixture = _Fixture({"delete_file": "ask"})
    result = fixture.call("delete_file", {"filepath": "TEST.txt"})
    assert _text(result) == "Successfully deleted TEST.txt"
    fixture.store.delete.assert_called_once_with("TEST.txt")


def test_given_an_allowed_command_when_called_then_it_is_only_simulated() -> None:
    fixture = _Fixture({"execute_command": "allow"})
    result = fixture.call("execute_command", {"command": "TEST_COMMAND"})
    assert "Simulated execution of command: TEST_COMMAND" in _text(result)
    assert "Actual execution disabled" in _text(result)


def test_given_the_default_deny_when_a_command_is_called_then_it_is_refused() -> None:
    result = _Fixture().call("execute_command", {"command": "TEST_COMMAND"})
    assert result.is_error
    assert _text(result) == "Permission denied for tool: execute_command"


def test_given_an_approved_write_when_it_finishes_then_every_step_is_audited_with_arguments() -> (
    None
):
    """Pins F-L47-7: one writer, and the entries carry what was asked, approved and run."""
    fixture = _Fixture()
    fixture.call("write_file", {"filepath": "TEST.txt", "content": "TEST_BODY"})
    entries = fixture.audit.append.call_args_list
    assert [entry.args[2] for entry in entries] == ["approval_requested", "approved", "succeeded"]
    assert all("TEST.txt" in entry.args[1] for entry in entries)


# --- resources and prompt ------------------------------------------------------------------


def _read_resource(fixture: _Fixture, uri: str) -> str:
    async def run() -> str:
        async with fixture.client() as client:
            result = await client.read_resource_mcp(uri)
            block = result.contents[0]
            assert isinstance(block, TextResourceContents)
            return block.text

    return asyncio.run(run())


def test_given_an_empty_audit_log_when_the_resource_is_read_then_it_says_so() -> None:
    assert _read_resource(_Fixture(), "file://audit/log") == "No audit log entries yet."


def test_given_audit_entries_when_the_resource_is_read_then_the_log_text_comes_back() -> None:
    fixture = _Fixture()
    fixture.audit.read_text.return_value = "TEST_LOG_LINE\n"
    assert _read_resource(fixture, "file://audit/log") == "TEST_LOG_LINE\n"


def test_given_the_policy_when_the_resource_is_read_then_effective_policies_come_back() -> None:
    fixture = _Fixture({"write_file": "allow"})
    policies = json.loads(_read_resource(fixture, "file://config/permissions"))
    assert policies["write_file"] == "allow"
    assert policies["delete_file"] == "deny"


def test_given_an_unreadable_policy_when_the_resource_is_read_then_it_is_an_error() -> None:
    fixture = _Fixture()
    fixture.policy_store.load_rules.side_effect = InvalidPolicyError("TEST_BAD_FILE")

    async def run() -> None:
        async with fixture.client() as client:
            await client.read_resource_mcp("file://config/permissions")

    with pytest.raises(Exception, match="TEST_BAD_FILE"):
        asyncio.run(run())


def _read_resource_text(fixture: _Fixture, uri: str, answer: Answer = "accept") -> str:
    async def run() -> str:
        async with fixture.client(answer) as client:
            result = await client.read_resource_mcp(uri)
            block = result.contents[0]
            assert isinstance(block, TextResourceContents)
            return block.text

    return asyncio.run(run())


@pytest.mark.parametrize(
    ("uri", "policy_name"),
    [
        ("file://audit/log", "read_audit_log"),
        ("file://config/permissions", "read_permissions"),
    ],
)
def test_given_a_deny_policy_for_a_resource_when_it_is_read_then_it_is_refused(
    uri: str, policy_name: str
) -> None:
    """Pins F-L47-9: the lab's resources had no policy; these follow the same rules as tools."""
    fixture = _Fixture({policy_name: "deny"})
    with pytest.raises(Exception, match=f"Permission denied for tool: {policy_name}"):
        _read_resource_text(fixture, uri)


def test_given_a_deny_policy_for_the_audit_log_when_it_is_read_then_the_log_is_not_touched() -> (
    None
):
    fixture = _Fixture({"read_audit_log": "deny"})
    with pytest.raises(Exception, match="Permission denied"):
        _read_resource_text(fixture, "file://audit/log")
    fixture.audit.read_text.assert_not_called()


def test_given_an_ask_policy_for_the_log_and_an_accept_when_read_then_the_log_comes_back() -> None:
    fixture = _Fixture({"read_audit_log": "ask"})
    fixture.audit.read_text.return_value = "TEST_LOG_LINE\n"
    assert _read_resource_text(fixture, "file://audit/log") == "TEST_LOG_LINE\n"
    assert len(fixture.questions) == 1
    assert "read_audit_log" in fixture.questions[0]


@pytest.mark.parametrize("answer", ["decline", "cancel"])
def test_given_an_ask_policy_for_the_audit_log_and_no_approval_when_read_then_it_is_refused(
    answer: Answer,
) -> None:
    fixture = _Fixture({"read_audit_log": "ask"})
    with pytest.raises(Exception, match="(declined|cancelled)"):
        _read_resource_text(fixture, "file://audit/log", answer)
    fixture.audit.read_text.assert_not_called()


def test_given_an_allowed_resource_read_when_it_finishes_then_the_read_is_audited() -> None:
    """Pins that reading the policy or the log leaves a trace in the log itself."""
    fixture = _Fixture({"read_audit_log": "allow"})
    _read_resource_text(fixture, "file://audit/log")
    entries = fixture.audit.append.call_args_list
    assert [(entry.args[0], entry.args[2]) for entry in entries] == [
        ("read_audit_log", "allowed"),
        ("read_audit_log", "succeeded"),
    ]


def test_given_the_security_review_prompt_when_fetched_then_it_names_the_operation_and_risk() -> (
    None
):
    async def run() -> str:
        async with _Fixture().client() as client:
            result = await client.get_prompt_mcp(
                "security_review", {"operation": "TEST_OPERATION", "risk_level": "HIGH"}
            )
            block = result.messages[0].content
            assert isinstance(block, TextContent)
            return block.text

    text = asyncio.run(run())
    assert "Operation: TEST_OPERATION" in text
    assert "Risk Level: HIGH" in text
    assert "What should be logged for audit purposes?" in text
