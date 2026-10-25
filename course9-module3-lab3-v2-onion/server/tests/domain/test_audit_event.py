from domain.audit_event import AuditEvent


def test_given_the_audit_events_when_listed_then_each_step_of_a_tool_call_has_one() -> None:
    """Pins F-L47-7: approvals and outcomes get their own events, not just the policy decision."""
    assert {event.value for event in AuditEvent} == {
        "allowed",
        "denied",
        "approval_requested",
        "approved",
        "declined",
        "cancelled",
        "succeeded",
        "failed",
    }
