from domain.approval_outcome import ApprovalOutcome


def test_given_the_outcomes_when_listed_then_they_are_accepted_declined_cancelled() -> None:
    """Pins the three-way answer to an elicitation; a fourth outcome has to break this test."""
    assert {outcome.value for outcome in ApprovalOutcome} == {"accepted", "declined", "cancelled"}


def test_given_each_outcome_when_asked_if_the_tool_runs_then_only_accepted_says_yes() -> None:
    """Pins the fail-closed rule: decline and cancel both stop the tool."""
    allowing = {outcome for outcome in ApprovalOutcome if outcome.lets_the_tool_run}
    assert allowing == {ApprovalOutcome.ACCEPTED}
