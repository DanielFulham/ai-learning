import pytest
from mcp_types import (
    CreateMessageResult,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputResponses,
    TextContent,
)

from application.input_required import APPROVAL_KEY, approval_request, outcome_from
from domain.approval_outcome import ApprovalOutcome


def _responses(result: ElicitResult) -> InputResponses:
    return {APPROVAL_KEY: result}


def test_given_a_question_when_the_request_is_built_then_it_is_one_elicitation_under_the_key() -> (
    None
):
    requests = approval_request("TEST_QUESTION")
    assert list(requests) == [APPROVAL_KEY]
    request = requests[APPROVAL_KEY]
    assert isinstance(request, ElicitRequest)
    assert request.method == "elicitation/create"


def test_given_a_question_when_the_request_is_built_then_it_asks_one_yes_or_no_field() -> None:
    """Pins that the form has one required boolean field, so clients render a working choice."""
    request = approval_request("TEST_QUESTION")[APPROVAL_KEY]
    assert isinstance(request, ElicitRequest)
    params = request.params
    assert isinstance(params, ElicitRequestFormParams)
    assert params.message == "TEST_QUESTION"
    assert params.requested_schema["properties"] == {"value": {"type": "boolean"}}
    assert params.requested_schema["required"] == ["value"]


def test_given_an_accept_with_yes_when_read_then_the_outcome_is_accepted() -> None:
    result = ElicitResult(action="accept", content={"value": True})
    assert outcome_from(_responses(result)) is ApprovalOutcome.ACCEPTED


@pytest.mark.parametrize("content", [{"value": False}, {}, None, {"value": "true"}, {"value": 1}])
def test_given_an_accept_that_is_not_a_clear_yes_when_read_then_the_outcome_is_declined(
    content: dict[str, str | int | float | bool | list[str] | None] | None,
) -> None:
    """Pins fail-closed: only a boolean true counts, never a truthy string or number."""
    result = ElicitResult(action="accept", content=content)
    assert outcome_from(_responses(result)) is ApprovalOutcome.DECLINED


def test_given_a_decline_when_read_then_the_outcome_is_declined() -> None:
    assert outcome_from(_responses(ElicitResult(action="decline"))) is ApprovalOutcome.DECLINED


def test_given_a_cancel_when_read_then_the_outcome_is_cancelled() -> None:
    assert outcome_from(_responses(ElicitResult(action="cancel"))) is ApprovalOutcome.CANCELLED


def test_given_no_responses_when_read_then_there_is_no_outcome() -> None:
    """Pins the first round: nothing answered yet means the tool has to ask."""
    assert outcome_from(None) is None


def test_given_responses_without_the_approval_key_when_read_then_there_is_no_outcome() -> None:
    other = ElicitResult(action="accept", content={"value": True})
    assert outcome_from({"TEST_OTHER_KEY": other}) is None


def test_given_an_answer_of_the_wrong_kind_when_read_then_the_outcome_is_cancelled() -> None:
    """Pins fail-closed: a response that is not an elicitation result never approves anything."""
    wrong = CreateMessageResult(
        role="assistant", content=TextContent(text="TEST_TEXT"), model="TEST_MODEL"
    )
    assert outcome_from({APPROVAL_KEY: wrong}) is ApprovalOutcome.CANCELLED
