from enum import StrEnum


class ApprovalChoice(StrEnum):
    """The user's answer to a question the server asked before running a tool."""

    ACCEPT = "accept"
    DECLINE = "decline"
    CANCEL = "cancel"
