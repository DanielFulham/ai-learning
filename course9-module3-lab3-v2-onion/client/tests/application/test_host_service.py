import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.host_service import HostService
from application.interfaces.approval_desk_interface import ApprovalDeskInterface
from domain.approval import ApprovalChoice
from domain.conversation import ChatReply, Message, ModelReply, ToolActivity, ToolCall
from interfaces.llm_chat_model_interface import LlmChatModelInterface
from interfaces.mcp_session_interface import (
    McpSessionInterface,
    RawPrompt,
    RawPromptMessage,
    RawPromptResult,
    RawResource,
    RawTool,
    RawToolResult,
)


def _make_session() -> MagicMock:
    session = MagicMock(spec=McpSessionInterface)
    session.list_tools.return_value = [
        RawTool(
            name="TEST_TOOL",
            description="TEST_DESC",
            input_schema={
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"],
            },
        )
    ]
    session.call_tool.return_value = RawToolResult(texts=["TEST_OUT"], is_error=False)
    session.list_resources.return_value = [
        RawResource(uri="file://TEST/one", name="TEST_NAME", description="TEST_RDESC")
    ]
    session.read_resource.return_value = ["TEST_RESOURCE_TEXT"]
    session.list_prompts.return_value = [
        RawPrompt(name="TEST_PROMPT", description="TEST_PDESC", argument_names=["topic"])
    ]
    session.get_prompt.return_value = RawPromptResult(
        description="TEST_PDESC", messages=[RawPromptMessage(role="user", text="TEST_PTEXT")]
    )
    return session


def _make_model(*replies: ModelReply | None) -> MagicMock:
    model = MagicMock(spec=LlmChatModelInterface)
    model.complete = AsyncMock(side_effect=list(replies))
    return model


def _make_desk(pending: str | None = None) -> MagicMock:
    desk = MagicMock(spec=ApprovalDeskInterface)
    desk.pending_message.return_value = pending
    desk.respond.return_value = True
    return desk


def _chat(
    service: HostService,
    text: str,
    history: tuple[Message, ...] = (),
    desk: MagicMock | None = None,
) -> ChatReply:
    return asyncio.run(service.chat(text, history, desk or _make_desk()))


def _call(name: str, arguments_json: str = "{}", call_id: str = "TEST_CALL") -> ToolCall:
    return ToolCall(call_id, name, arguments_json)


# --- the desk: answering a waiting question -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "choice"),
    [
        ("yes", ApprovalChoice.ACCEPT),
        ("  Y  ", ApprovalChoice.ACCEPT),
        ("no", ApprovalChoice.DECLINE),
        ("deny", ApprovalChoice.DECLINE),
        ("cancel", ApprovalChoice.CANCEL),
    ],
)
def test_given_a_waiting_question_when_the_user_types_a_reply_word_then_the_desk_gets_that_choice(
    text: str, choice: ApprovalChoice
) -> None:
    desk, model = _make_desk("TEST_QUESTION"), _make_model()
    _chat(HostService(_make_session(), model), text, desk=desk)
    desk.respond.assert_called_once_with(choice)
    model.complete.assert_not_called()


def test_given_a_waiting_question_when_answered_then_the_reply_says_what_was_sent() -> None:
    desk = _make_desk("TEST_QUESTION")
    reply = _chat(HostService(_make_session(), _make_model()), "yes", desk=desk)
    assert "accept" in reply.text.lower()
    assert reply.tool_activity == ()


def test_given_a_waiting_question_when_answered_then_the_history_is_returned_unchanged() -> None:
    history = (Message(role="user", text="TEST_EARLIER"),)
    desk = _make_desk("TEST_QUESTION")
    reply = _chat(HostService(_make_session(), _make_model()), "yes", history, desk)
    assert reply.history == history


def test_given_a_waiting_question_when_the_user_types_anything_else_then_it_is_repeated() -> None:
    """Pins that a sentence never answers a question, and the user is shown what is waiting."""
    desk, model = _make_desk("TEST_QUESTION"), _make_model()
    reply = _chat(HostService(_make_session(), model), "go ahead and write it", desk=desk)
    assert "TEST_QUESTION" in reply.text
    desk.respond.assert_not_called()
    model.complete.assert_not_called()


def test_given_nothing_waiting_when_the_user_types_yes_then_it_is_an_ordinary_message() -> None:
    """Pins that a stray "yes" never approves anything: with no question it goes to the model."""
    desk = _make_desk(None)
    model = _make_model(ModelReply(text="TEST_ANSWER"))
    reply = _chat(HostService(_make_session(), model), "yes", desk=desk)
    desk.respond.assert_not_called()
    assert reply.text == "TEST_ANSWER"


def test_given_a_question_that_was_just_answered_elsewhere_when_replied_then_it_is_ordinary() -> (
    None
):
    desk = _make_desk("TEST_QUESTION")
    desk.respond.return_value = False
    model = _make_model(ModelReply(text="TEST_ANSWER"))
    assert _chat(HostService(_make_session(), model), "yes", desk=desk).text == "TEST_ANSWER"


# --- the tools the model is offered ---------------------------------------------------------


def test_given_a_server_when_the_tools_are_collected_then_the_schema_is_carried_across() -> None:
    tools = asyncio.run(HostService(_make_session(), _make_model()).get_available_tools())
    first = tools[0]
    assert first.name == "TEST_TOOL"
    assert first.input_schema == {
        "type": "object",
        "properties": {"filepath": {"type": "string"}},
        "required": ["filepath"],
    }


def test_given_a_tool_with_no_description_when_collected_then_a_default_is_used() -> None:
    session = _make_session()
    session.list_tools.return_value = [RawTool(name="TEST_TOOL", description="", input_schema={})]
    tools = asyncio.run(HostService(session, _make_model()).get_available_tools())
    assert tools[0].description == "Execute TEST_TOOL"
    assert tools[0].input_schema == {"type": "object", "properties": {}}


def test_given_a_server_when_the_tools_are_collected_then_four_helper_tools_follow() -> None:
    tools = asyncio.run(HostService(_make_session(), _make_model()).get_available_tools())
    assert [tool.name for tool in tools[1:]] == [
        "mcp_list_resources",
        "mcp_read_resource",
        "mcp_list_prompts",
        "mcp_get_prompt",
    ]


# --- running a tool call --------------------------------------------------------------------


def _execute(
    service: HostService,
    name: str,
    arguments: dict[str, object],
    desk: MagicMock | None = None,
) -> str:
    return asyncio.run(service.execute_tool(name, arguments, desk or _make_desk()))


def test_given_a_server_tool_when_run_then_the_calling_sessions_desk_is_passed_along() -> None:
    """Pins F-L47-18: the call carries this chat session's own desk to the session."""
    session, desk = _make_session(), _make_desk()
    _execute(HostService(session, _make_model()), "TEST_TOOL", {"filepath": "a"}, desk)
    session.call_tool.assert_awaited_once_with("TEST_TOOL", {"filepath": "a"}, desk)


def test_given_a_server_tool_when_run_then_its_first_text_comes_back() -> None:
    assert _execute(HostService(_make_session(), _make_model()), "TEST_TOOL", {}) == "TEST_OUT"


def test_given_a_flagged_tool_error_when_run_then_the_model_gets_the_error_text() -> None:
    session = _make_session()
    session.call_tool.return_value = RawToolResult(texts=["TEST_BAD"], is_error=True)
    text = _execute(HostService(session, _make_model()), "TEST_TOOL", {})
    assert text == "Error executing tool: TEST_BAD"


def test_given_a_session_that_raises_when_a_tool_is_run_then_the_model_gets_the_message() -> None:
    session = _make_session()
    session.call_tool.side_effect = RuntimeError("TEST_BOOM")
    text = _execute(HostService(session, _make_model()), "TEST_TOOL", {})
    assert text == "Error executing tool: TEST_BOOM"


def test_given_the_list_resources_helper_when_run_then_each_resource_is_listed() -> None:
    text = _execute(HostService(_make_session(), _make_model()), "mcp_list_resources", {})
    assert text == "Available resources:\n- file://TEST/one (TEST_NAME): TEST_RDESC\n"


def test_given_the_read_resource_helper_when_run_then_the_text_comes_back_with_the_desk() -> None:
    session, desk = _make_session(), _make_desk()
    service = HostService(session, _make_model())
    text = _execute(service, "mcp_read_resource", {"uri": "file://TEST/one"}, desk)
    assert text == "TEST_RESOURCE_TEXT"
    session.read_resource.assert_awaited_once_with("file://TEST/one", desk)


@pytest.mark.parametrize("arguments", [{}, {"uri": ""}, {"uri": 5}])
def test_given_the_read_resource_helper_with_no_usable_uri_when_run_then_it_is_refused(
    arguments: dict[str, object],
) -> None:
    session = _make_session()
    text = _execute(HostService(session, _make_model()), "mcp_read_resource", arguments)
    assert text == "Error: URI is required"
    session.read_resource.assert_not_called()


def test_given_a_refused_resource_read_when_the_helper_runs_then_the_model_gets_the_reason() -> (
    None
):
    session = _make_session()
    session.read_resource.side_effect = RuntimeError("Permission denied for tool: read_audit_log")
    text = _execute(HostService(session, _make_model()), "mcp_read_resource", {"uri": "u"})
    assert text == "Error reading resource: Permission denied for tool: read_audit_log"


def test_given_the_list_prompts_helper_when_run_then_each_prompt_is_listed_with_its_args() -> None:
    text = _execute(HostService(_make_session(), _make_model()), "mcp_list_prompts", {})
    assert text == "Available prompts:\n- TEST_PROMPT: TEST_PDESC (args: topic)\n"


def test_given_the_get_prompt_helper_with_nested_arguments_when_run_then_they_are_used() -> None:
    session = _make_session()
    arguments: dict[str, object] = {"name": "TEST_PROMPT", "arguments": {"topic": "T"}}
    text = _execute(HostService(session, _make_model()), "mcp_get_prompt", arguments)
    assert text == "Prompt: TEST_PROMPT\n\n[user]: TEST_PTEXT\n\n"
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"topic": "T"})


def test_given_the_get_prompt_helper_with_arguments_beside_the_name_when_run_then_they_work() -> (
    None
):
    """Pins F-L46-5's fix: a model may put the arguments beside `name`, as the lab's text says."""
    session = _make_session()
    arguments: dict[str, object] = {"name": "TEST_PROMPT", "topic": "T"}
    _execute(HostService(session, _make_model()), "mcp_get_prompt", arguments)
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"topic": "T"})


def test_given_the_get_prompt_helper_with_no_name_when_run_then_it_is_refused() -> None:
    session = _make_session()
    text = _execute(HostService(session, _make_model()), "mcp_get_prompt", {})
    assert text == "Error: Prompt name is required"
    session.get_prompt.assert_not_called()


# --- the chat -------------------------------------------------------------------------------


def test_given_a_text_reply_when_chatting_then_the_answer_and_the_new_history_come_back() -> None:
    model = _make_model(ModelReply(text="TEST_ANSWER"))
    reply = _chat(HostService(_make_session(), model), "TEST_QUESTION")
    assert reply.text == "TEST_ANSWER"
    assert reply.history == (
        Message(role="user", text="TEST_QUESTION"),
        Message(role="assistant", text="TEST_ANSWER"),
    )


def test_given_a_history_when_chatting_then_it_is_sent_to_the_model_before_the_new_message() -> (
    None
):
    model = _make_model(ModelReply(text="TEST_ANSWER"))
    history = (Message(role="user", text="TEST_EARLIER"),)
    _chat(HostService(_make_session(), model), "TEST_NOW", history)
    sent = model.complete.await_args.args[0]
    assert [m.text for m in sent] == ["TEST_EARLIER", "TEST_NOW"]


def test_given_two_chats_with_different_histories_when_run_then_neither_sees_the_other() -> None:
    """Pins F-L46-18's fix: the service keeps no conversation, so sessions cannot share one."""
    model = _make_model(ModelReply(text="TEST_A"), ModelReply(text="TEST_B"))
    service = HostService(_make_session(), model)
    first = _chat(service, "TEST_ONE", (Message(role="user", text="TEST_SESSION_A"),))
    second = _chat(service, "TEST_TWO", (Message(role="user", text="TEST_SESSION_B"),))
    assert "TEST_SESSION_A" not in [m.text for m in second.history]
    assert "TEST_SESSION_B" not in [m.text for m in first.history]


def test_given_a_tool_call_when_chatting_then_the_tool_runs_and_the_final_answer_follows() -> None:
    call = _call("TEST_TOOL", '{"filepath": "a"}')
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), ModelReply(text="TEST_FINAL"))
    session = _make_session()
    reply = _chat(HostService(session, model), "TEST_QUESTION")
    assert reply.text == "TEST_FINAL"
    assert reply.tool_activity == (ToolActivity("TEST_TOOL", '{"filepath": "a"}', "TEST_OUT"),)
    assert [m.role for m in reply.history] == ["user", "assistant", "tool", "assistant"]


def test_given_a_tool_call_when_chatting_then_the_sessions_desk_reaches_the_tool_call() -> None:
    call = _call("TEST_TOOL", "{}")
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), ModelReply(text="TEST_FINAL"))
    session, desk = _make_session(), _make_desk()
    _chat(HostService(session, model), "TEST_QUESTION", desk=desk)
    assert session.call_tool.await_args.args[2] is desk


def test_given_a_tool_round_when_the_final_call_is_made_then_no_tools_are_offered() -> None:
    call = _call("TEST_TOOL", "{}")
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), ModelReply(text="TEST_FINAL"))
    _chat(HostService(_make_session(), model), "TEST_QUESTION")
    assert model.complete.await_args_list[1].args[1] == []


@pytest.mark.parametrize(
    ("arguments_json", "expected"),
    [
        ("{not json", "Error: the tool arguments were not valid JSON"),
        ("[1, 2]", "Error: the tool arguments must be a JSON object"),
    ],
)
def test_given_bad_tool_arguments_when_chatting_then_the_model_is_told_and_nothing_runs(
    arguments_json: str, expected: str
) -> None:
    """Lab 46 fix kept: bad arguments go back to the model as the tool's result, not a crash."""
    call = _call("TEST_TOOL", arguments_json)
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), ModelReply(text="TEST_FINAL"))
    session = _make_session()
    reply = _chat(HostService(session, model), "TEST_QUESTION")
    assert reply.tool_activity[0].result.startswith(expected)
    session.call_tool.assert_not_called()


def test_given_no_reply_from_the_model_when_chatting_then_an_error_text_comes_back() -> None:
    reply = _chat(HostService(_make_session(), _make_model(None)), "TEST_QUESTION")
    assert reply.text == "Error: No response from LLM"


def test_given_no_final_reply_after_the_tools_when_chatting_then_an_error_text_comes_back() -> None:
    call = _call("TEST_TOOL", "{}")
    model = _make_model(ModelReply(text=None, tool_calls=(call,)), None)
    reply = _chat(HostService(_make_session(), model), "TEST_QUESTION")
    assert reply.text == "Error: No response from LLM after tool execution"
    assert len(reply.tool_activity) == 1


def test_given_an_empty_model_answer_when_chatting_then_it_is_never_none_or_blank() -> None:
    reply = _chat(HostService(_make_session(), _make_model(ModelReply(text=None))), "TEST")
    assert reply.text == "The model returned no text."
