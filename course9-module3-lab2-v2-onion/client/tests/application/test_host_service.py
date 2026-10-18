import asyncio
import json
from unittest.mock import MagicMock

import pytest

from application.host_service import HostService
from domain.conversation import ChatReply, ModelReply, ToolActivity, ToolCall
from interfaces.llm_chat_model_interface import LlmChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface

_SYNTHETIC_NAMES = ["mcp_list_resources", "mcp_read_resource", "mcp_list_prompts", "mcp_get_prompt"]


def _make_session() -> MagicMock:
    session = MagicMock(spec=McpSessionInterface)
    session.list_tools.return_value = [
        {
            "name": "TEST_TOOL",
            "description": "TEST_DESC",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "title": "TEST_TITLE",
            },
        }
    ]
    session.call_tool.return_value = {"texts": ["TEST_RESULT", "TEST_SECOND"], "is_error": False}
    session.list_resource_templates.return_value = [
        {"name": "TEST_NAME", "uri_template": "TEST://{x}", "description": "TEST_DOC"}
    ]
    session.read_resource.return_value = ["TEST_FILE_TEXT"]
    session.list_prompts.return_value = [
        {"name": "TEST_PROMPT", "description": "TEST_DESC", "argument_names": ["TEST_ARG"]}
    ]
    session.get_prompt.return_value = {
        "description": "TEST_RENDERED",
        "messages": [{"role": "user", "text": "TEST_MESSAGE"}],
    }
    return session


def _make_model(*replies: ModelReply | None) -> MagicMock:
    model = MagicMock(spec=LlmChatModelInterface)
    model.complete.side_effect = list(replies)
    return model


def _service(session: MagicMock, model: MagicMock | None = None) -> HostService:
    return HostService(session, model or _make_model())


def _execute(service: HostService, tool: str, arguments: dict[str, object]) -> str:
    return asyncio.run(service.execute_tool(tool, arguments))


def test_the_server_tools_come_first_then_the_four_helper_tools() -> None:
    tools = asyncio.run(_service(_make_session()).get_available_tools())
    assert [t.name for t in tools] == ["TEST_TOOL", *_SYNTHETIC_NAMES]


def test_a_server_tool_keeps_its_properties_and_required_and_drops_the_rest_of_the_schema() -> None:
    tool = asyncio.run(_service(_make_session()).get_available_tools())[0]
    assert tool.description == "TEST_DESC"
    assert tool.input_schema == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }


def test_a_server_tool_with_no_description_or_properties_gets_the_lab_defaults() -> None:
    session = _make_session()
    session.list_tools.return_value = [{"name": "TEST_BARE", "description": "", "input_schema": {}}]
    tool = asyncio.run(_service(session).get_available_tools())[0]
    assert tool.description == "Execute TEST_BARE"
    assert tool.input_schema == {"type": "object", "properties": {}}


def test_empty_required_is_left_out() -> None:
    session = _make_session()
    session.list_tools.return_value = [
        {"name": "T", "description": "D", "input_schema": {"properties": {}, "required": []}}
    ]
    assert "required" not in asyncio.run(_service(session).get_available_tools())[0].input_schema


def test_the_helper_tools_have_the_labs_parameters() -> None:
    tools = {t.name: t for t in asyncio.run(_service(_make_session()).get_available_tools())}
    assert tools["mcp_read_resource"].input_schema["required"] == ["uri"]
    assert tools["mcp_get_prompt"].input_schema["required"] == ["name"]
    properties = tools["mcp_get_prompt"].input_schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"name", "arguments"}
    assert tools["mcp_list_resources"].input_schema == {"type": "object", "properties": {}}


def test_list_resources_shows_the_template_name_and_description() -> None:
    text = _execute(_service(_make_session()), "mcp_list_resources", {})
    assert text == "Available resources:\n- TEST://{x} (TEST_NAME): TEST_DOC\n"


def test_read_resource_returns_the_files_text_not_a_repr() -> None:
    session = _make_session()
    text = _execute(_service(session), "mcp_read_resource", {"uri": "file://workspace/a.txt"})
    session.read_resource.assert_awaited_once_with("file://workspace/a.txt")
    assert text == "TEST_FILE_TEXT"


def test_read_resource_without_a_uri_is_an_error() -> None:
    session = _make_session()
    assert _execute(_service(session), "mcp_read_resource", {}) == "Error: URI is required"
    session.read_resource.assert_not_called()


def test_read_resource_failure_is_reported_as_text() -> None:
    session = _make_session()
    session.read_resource.side_effect = RuntimeError("TEST_BOOM")
    text = _execute(_service(session), "mcp_read_resource", {"uri": "TEST://x"})
    assert text == "Error reading resource: TEST_BOOM"


def test_list_prompts_shows_the_arguments() -> None:
    text = _execute(_service(_make_session()), "mcp_list_prompts", {})
    assert text == "Available prompts:\n- TEST_PROMPT: TEST_DESC (args: TEST_ARG)\n"


def test_get_prompt_shows_each_message_with_its_role_not_unknown() -> None:
    session = _make_session()
    arguments: dict[str, object] = {"name": "TEST_PROMPT", "arguments": {"TEST_ARG": "V"}}
    text = _execute(_service(session), "mcp_get_prompt", arguments)
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"TEST_ARG": "V"})
    assert text == "Prompt: TEST_PROMPT\n\n[user]: TEST_MESSAGE\n\n"


def test_get_prompt_accepts_the_arguments_beside_the_name() -> None:
    # The lab's text says the model passes `name` and `filename` side by side, but its code reads
    # only a nested `arguments` object, so the server was called with none.
    session = _make_session()
    _execute(_service(session), "mcp_get_prompt", {"name": "TEST_PROMPT", "filename": "a.txt"})
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"filename": "a.txt"})


def test_get_prompt_prefers_a_filled_arguments_object_over_the_loose_keys() -> None:
    session = _make_session()
    arguments: dict[str, object] = {
        "name": "TEST_PROMPT",
        "arguments": {"filename": "nested.txt"},
        "filename": "loose.txt",
    }
    _execute(_service(session), "mcp_get_prompt", arguments)
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"filename": "nested.txt"})


def test_get_prompt_with_an_empty_arguments_object_uses_the_loose_keys() -> None:
    session = _make_session()
    arguments: dict[str, object] = {"name": "TEST_PROMPT", "arguments": {}, "filename": "a.txt"}
    _execute(_service(session), "mcp_get_prompt", arguments)
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"filename": "a.txt"})


def test_get_prompt_with_no_arguments_at_all_sends_none() -> None:
    session = _make_session()
    _execute(_service(session), "mcp_get_prompt", {"name": "TEST_PROMPT"})
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {})


def test_the_get_prompt_tool_tells_the_model_what_goes_in_arguments() -> None:
    tools = {t.name: t for t in asyncio.run(_service(_make_session()).get_available_tools())}
    properties = tools["mcp_get_prompt"].input_schema["properties"]
    assert isinstance(properties, dict)
    description = properties["arguments"]["description"]
    assert "filename" in description


def test_get_prompt_without_a_name_is_an_error() -> None:
    session = _make_session()
    assert _execute(_service(session), "mcp_get_prompt", {}) == "Error: Prompt name is required"
    session.get_prompt.assert_not_called()


def test_get_prompt_failure_is_reported_as_text() -> None:
    session = _make_session()
    session.get_prompt.side_effect = RuntimeError("TEST_BOOM")
    text = _execute(_service(session), "mcp_get_prompt", {"name": "TEST_PROMPT"})
    assert text == "Error getting prompt: TEST_BOOM"


def test_a_server_tool_returns_its_first_text_part_not_a_repr() -> None:
    session = _make_session()
    assert _execute(_service(session), "TEST_TOOL", {"a": "V"}) == "TEST_RESULT"
    session.call_tool.assert_awaited_once_with("TEST_TOOL", {"a": "V"})


def test_a_server_tool_that_returns_no_text_gives_an_empty_string() -> None:
    session = _make_session()
    session.call_tool.return_value = {"texts": [], "is_error": False}
    assert _execute(_service(session), "TEST_TOOL", {}) == ""


def test_a_server_tool_failure_is_reported_as_text() -> None:
    session = _make_session()
    session.call_tool.side_effect = RuntimeError("TEST_BOOM")
    assert _execute(_service(session), "TEST_TOOL", {}) == "Error executing tool: TEST_BOOM"


def test_a_result_the_server_flagged_as_an_error_is_reported_to_the_model_as_an_error() -> None:
    # The lab ignores the error flag, so the model could not tell a failure from an answer.
    session = _make_session()
    session.call_tool.return_value = {"texts": ["TEST_BOOM"], "is_error": True}
    assert _execute(_service(session), "TEST_TOOL", {}) == "Error executing tool: TEST_BOOM"


def test_an_error_result_with_no_text_still_says_so_to_the_model() -> None:
    session = _make_session()
    session.call_tool.return_value = {"texts": [], "is_error": True}
    text = _execute(_service(session), "TEST_TOOL", {})
    assert text == "Error executing tool: the tool reported an error"


def test_chat_without_tool_calls_returns_the_text_and_keeps_the_history() -> None:
    session = _make_session()
    model = _make_model(ModelReply(text="TEST_ANSWER"))
    service = _service(session, model)
    assert asyncio.run(service.chat("TEST_QUERY")) == ChatReply("TEST_ANSWER")
    session.connect.assert_awaited()
    assert [(m.role, m.text) for m in service.history] == [
        ("user", "TEST_QUERY"),
        ("assistant", "TEST_ANSWER"),
    ]
    offered = [t.name for t in model.complete.call_args.args[1]]
    assert offered == ["TEST_TOOL", *_SYNTHETIC_NAMES]


def test_chat_runs_the_tool_the_model_asks_for_then_asks_again_with_no_tools() -> None:
    session = _make_session()
    call = ToolCall("TEST_CALL", "TEST_TOOL", json.dumps({"a": "V"}))
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), ModelReply(text="TEST_FINAL"))
    service = _service(session, model)
    reply = asyncio.run(service.chat("TEST_QUERY"))
    assert reply == ChatReply(
        "TEST_FINAL", (ToolActivity("TEST_TOOL", json.dumps({"a": "V"}), "TEST_RESULT"),)
    )
    session.call_tool.assert_awaited_once_with("TEST_TOOL", {"a": "V"})
    assert model.complete.call_args_list[1].args[1] == []
    assert [(m.role, m.text, m.tool_call_id) for m in service.history] == [
        ("user", "TEST_QUERY", None),
        ("assistant", "", None),
        ("tool", "TEST_RESULT", "TEST_CALL"),
        ("assistant", "TEST_FINAL", None),
    ]
    assert service.history[1].tool_calls == (call,)


def test_chat_stops_after_one_round_of_tool_calls() -> None:
    call = ToolCall("TEST_CALL", "TEST_TOOL", "{}")
    model = _make_model(
        ModelReply(text=None, tool_calls=(call,)),
        ModelReply(text=None, tool_calls=(call,)),
    )
    service = _service(_make_session(), model)
    asyncio.run(service.chat("TEST_QUERY"))
    assert model.complete.await_count == 2


def test_chat_with_no_answer_from_the_model_says_so() -> None:
    reply = asyncio.run(_service(_make_session(), _make_model(None)).chat("Q"))
    assert reply == ChatReply("Error: No response from LLM")


def test_chat_with_no_answer_after_the_tools_says_so() -> None:
    call = ToolCall("TEST_CALL", "TEST_TOOL", "{}")
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), None)
    reply = asyncio.run(_service(_make_session(), model).chat("Q"))
    assert reply.text == "Error: No response from LLM after tool execution"
    assert [a.name for a in reply.tool_activity] == ["TEST_TOOL"]


def test_tool_arguments_that_are_not_json_are_reported_to_the_model_not_raised() -> None:
    # The lab's json.loads is unguarded, so one malformed answer from the model crashes the chat.
    session = _make_session()
    call = ToolCall("TEST_CALL", "TEST_TOOL", "not json")
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), ModelReply(text="TEST_FINAL"))
    service = _service(session, model)
    reply = asyncio.run(service.chat("Q"))
    assert reply.text == "TEST_FINAL"
    assert reply.tool_activity[0].result.startswith("Error: the tool arguments were not valid JSON")
    session.call_tool.assert_not_called()
    tool_message = service.history[2]
    assert (tool_message.role, tool_message.tool_call_id) == ("tool", "TEST_CALL")
    assert (tool_message.text or "").startswith("Error: the tool arguments were not valid JSON")


@pytest.mark.parametrize("arguments_json", ["[1, 2]", '"text"', "3", "null"])
def test_tool_arguments_that_are_json_but_not_an_object_are_reported_to_the_model(
    arguments_json: str,
) -> None:
    session = _make_session()
    call = ToolCall("TEST_CALL", "TEST_TOOL", arguments_json)
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), ModelReply(text="TEST_FINAL"))
    service = _service(session, model)
    assert asyncio.run(service.chat("Q")).text == "TEST_FINAL"
    session.call_tool.assert_not_called()
    assert service.history[2].text == "Error: the tool arguments must be a JSON object"


def test_reset_clears_the_history() -> None:
    service = _service(_make_session(), _make_model(ModelReply(text="TEST_ANSWER")))
    asyncio.run(service.chat("TEST_QUERY"))
    service.reset()
    assert service.history == ()


@pytest.mark.parametrize("text", [None, ""])
def test_a_model_answer_with_no_text_is_replaced_by_a_clear_message(text: str | None) -> None:
    service = _service(_make_session(), _make_model(ModelReply(text=text)))
    assert asyncio.run(service.chat("Q")).text == "The model returned no text."


def test_a_final_answer_with_no_text_after_the_tools_is_replaced_by_a_clear_message() -> None:
    call = ToolCall("TEST_CALL", "TEST_TOOL", "{}")
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), ModelReply(text=None))
    reply = asyncio.run(_service(_make_session(), model).chat("Q"))
    assert reply.text == "The model returned no text."
    assert len(reply.tool_activity) == 1
