from enum import StrEnum


class AuditEvent(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVED = "approved"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
