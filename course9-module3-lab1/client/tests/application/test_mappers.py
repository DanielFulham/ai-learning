from application.mappers import (
    to_prompt_info,
    to_rendered_prompt,
    to_resource_template_info,
    to_tool_call_result,
    to_tool_info,
)
from interfaces.mcp_session_interface import (
    RawPrompt,
    RawPromptResult,
    RawResourceTemplate,
    RawTool,
    RawToolResult,
)


def test_to_tool_info_maps_name_and_description() -> None:
    raw: RawTool = {"name": "TEST_TOOL", "description": "TEST_DESC"}
    tool = to_tool_info(raw)
    assert (tool.name, tool.description) == ("TEST_TOOL", "TEST_DESC")


def test_to_tool_call_result_keeps_the_texts_in_order_as_a_tuple() -> None:
    raw: RawToolResult = {"texts": ["TEST_ONE", "TEST_TWO"], "is_error": False}
    result = to_tool_call_result(raw)
    assert result.texts == ("TEST_ONE", "TEST_TWO")
    assert result.is_error is False


def test_to_tool_call_result_keeps_the_error_flag() -> None:
    raw: RawToolResult = {"texts": ["TEST_BOOM"], "is_error": True}
    assert to_tool_call_result(raw).is_error is True


def test_to_resource_template_info_maps_name_and_uri_template() -> None:
    raw: RawResourceTemplate = {"name": "TEST_NAME", "uri_template": "TEST://{x}"}
    template = to_resource_template_info(raw)
    assert (template.name, template.uri_template) == ("TEST_NAME", "TEST://{x}")


def test_to_prompt_info_maps_argument_names_as_a_tuple() -> None:
    raw: RawPrompt = {
        "name": "TEST_PROMPT",
        "description": "TEST_DESC",
        "argument_names": ["TEST_ARG_A", "TEST_ARG_B"],
    }
    prompt = to_prompt_info(raw)
    assert (prompt.name, prompt.description) == ("TEST_PROMPT", "TEST_DESC")
    assert prompt.argument_names == ("TEST_ARG_A", "TEST_ARG_B")


def test_to_prompt_info_with_no_arguments_gives_an_empty_tuple() -> None:
    raw: RawPrompt = {"name": "TEST_PROMPT", "description": "", "argument_names": []}
    assert to_prompt_info(raw).argument_names == ()


def test_to_rendered_prompt_maps_description_and_messages_in_order() -> None:
    raw: RawPromptResult = {
        "description": "TEST_DESC",
        "messages": [
            {"role": "user", "text": "TEST_FIRST"},
            {"role": "assistant", "text": "TEST_SECOND"},
        ],
    }
    prompt = to_rendered_prompt(raw)
    assert prompt.description == "TEST_DESC"
    assert [(m.role, m.text) for m in prompt.messages] == [
        ("user", "TEST_FIRST"),
        ("assistant", "TEST_SECOND"),
    ]
