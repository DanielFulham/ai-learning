from application.approval_broker import ApprovalBroker
from application.approval_desks import ApprovalDesks
from application.interfaces.approval_desk_interface import ApprovalDeskInterface


def test_given_a_broker_when_typed_as_a_desk_then_it_satisfies_the_interface() -> None:
    desk: ApprovalDeskInterface = ApprovalBroker()
    assert desk is not None


def test_given_the_same_session_id_when_a_desk_is_asked_for_twice_then_it_is_the_same_desk() -> (
    None
):
    desks = ApprovalDesks()
    assert desks.desk_for("TEST_SESSION_A") is desks.desk_for("TEST_SESSION_A")


def test_given_two_session_ids_when_desks_are_asked_for_then_they_are_different_desks() -> None:
    """Pins F-L47-18: a "yes" in one browser session has no way to reach another's question."""
    desks = ApprovalDesks()
    assert desks.desk_for("TEST_SESSION_A") is not desks.desk_for("TEST_SESSION_B")


def test_given_a_forgotten_session_when_its_desk_is_asked_for_then_a_new_one_is_made() -> None:
    desks = ApprovalDesks()
    first = desks.desk_for("TEST_SESSION_A")
    desks.forget("TEST_SESSION_A")
    assert desks.desk_for("TEST_SESSION_A") is not first


def test_given_an_unknown_session_when_forgotten_then_nothing_breaks() -> None:
    ApprovalDesks().forget("TEST_NEVER_SEEN")


def test_given_desks_when_sessions_come_and_go_then_the_count_follows() -> None:
    desks = ApprovalDesks()
    assert desks.count() == 0
    desks.desk_for("TEST_SESSION_A")
    desks.desk_for("TEST_SESSION_B")
    desks.desk_for("TEST_SESSION_A")
    assert desks.count() == 2
    desks.forget("TEST_SESSION_A")
    assert desks.count() == 1
