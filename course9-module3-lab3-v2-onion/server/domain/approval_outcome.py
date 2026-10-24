from enum import StrEnum


class ApprovalOutcome(StrEnum):
    ACCEPTED = "accepted"
    DECLINED = "declined"
    CANCELLED = "cancelled"

    @property
    def lets_the_tool_run(self) -> bool:
        return self is ApprovalOutcome.ACCEPTED
