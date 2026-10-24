from mcp_types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequests,
    InputResponses,
)

from domain.approval_outcome import ApprovalOutcome

APPROVAL_KEY = "approval"


def approval_request(message: str) -> InputRequests:
    """Build the one yes-or-no elicitation a tool returns when the policy says ask."""
    return {
        APPROVAL_KEY: ElicitRequest(
            method="elicitation/create",
            params=ElicitRequestFormParams(
                message=message,
                requested_schema={
                    "type": "object",
                    "properties": {"value": {"type": "boolean"}},
                    "required": ["value"],
                },
            ),
        )
    }


def outcome_from(responses: InputResponses | None) -> ApprovalOutcome | None:
    """Read the client's answer. None means nothing was answered yet, so the tool has to ask."""
    if responses is None or APPROVAL_KEY not in responses:
        return None
    answer = responses[APPROVAL_KEY]
    if not isinstance(answer, ElicitResult):
        return ApprovalOutcome.CANCELLED
    if answer.action == "accept":
        # Only a boolean true counts: a truthy string or number is not a clear yes.
        yes = answer.content is not None and answer.content.get("value") is True
        return ApprovalOutcome.ACCEPTED if yes else ApprovalOutcome.DECLINED
    if answer.action == "decline":
        return ApprovalOutcome.DECLINED
    return ApprovalOutcome.CANCELLED
