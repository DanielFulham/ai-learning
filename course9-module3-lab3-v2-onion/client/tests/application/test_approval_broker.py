import asyncio

import pytest

from application.approval_broker import ApprovalBroker
from domain.approval import ApprovalChoice
from interfaces.approval_source_interface import ApprovalAnswer, ApprovalSourceInterface


async def _let_tasks_run() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


def test_given_a_broker_when_typed_as_an_approval_source_then_it_satisfies_the_interface() -> None:
    source: ApprovalSourceInterface = ApprovalBroker()
    assert source is not None


def test_given_no_question_when_asked_for_the_pending_one_then_there_is_none() -> None:
    assert ApprovalBroker().pending_message() is None


def test_given_a_question_when_decided_then_it_waits_and_is_the_pending_message() -> None:
    async def run() -> tuple[str | None, bool]:
        broker = ApprovalBroker()
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        pending, finished = broker.pending_message(), task.done()
        broker.respond(ApprovalChoice.CANCEL)
        await task
        return pending, finished

    assert asyncio.run(run()) == ("TEST_QUESTION", False)


@pytest.mark.parametrize(
    ("choice", "answer"),
    [
        (ApprovalChoice.ACCEPT, "accept"),
        (ApprovalChoice.DECLINE, "decline"),
        (ApprovalChoice.CANCEL, "cancel"),
    ],
)
def test_given_a_waiting_question_when_the_user_responds_then_decide_returns_that_answer(
    choice: ApprovalChoice, answer: ApprovalAnswer
) -> None:
    async def run() -> ApprovalAnswer:
        broker = ApprovalBroker()
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        assert broker.respond(choice) is True
        return await task

    assert asyncio.run(run()) == answer


def test_given_a_question_that_was_answered_when_asked_for_the_pending_one_then_there_is_none() -> (
    None
):
    async def run() -> str | None:
        broker = ApprovalBroker()
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        broker.respond(ApprovalChoice.ACCEPT)
        await task
        return broker.pending_message()

    assert asyncio.run(run()) is None


def test_given_nothing_waiting_when_the_user_responds_then_nothing_happens() -> None:
    """Pins that a stray click or a stray "yes" never approves something that was not asked."""
    broker = ApprovalBroker()
    assert broker.respond(ApprovalChoice.ACCEPT) is False


def test_given_one_answer_when_the_user_responds_twice_then_only_the_first_counts() -> None:
    async def run() -> tuple[bool, bool]:
        broker = ApprovalBroker()
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        first = broker.respond(ApprovalChoice.DECLINE)
        second = broker.respond(ApprovalChoice.ACCEPT)
        assert await task == "decline"
        return first, second

    assert asyncio.run(run()) == (True, False)


def test_given_two_waiting_questions_when_the_user_responds_then_the_oldest_is_answered_first() -> (
    None
):
    async def run() -> tuple[ApprovalAnswer, str | None]:
        broker = ApprovalBroker()
        first = asyncio.create_task(broker.decide("TEST_FIRST"))
        await _let_tasks_run()
        second = asyncio.create_task(broker.decide("TEST_SECOND"))
        await _let_tasks_run()
        broker.respond(ApprovalChoice.ACCEPT)
        answer = await first
        pending = broker.pending_message()
        broker.respond(ApprovalChoice.CANCEL)
        await second
        return answer, pending

    assert asyncio.run(run()) == ("accept", "TEST_SECOND")


def test_given_no_response_when_the_timeout_passes_then_the_answer_is_cancel() -> None:
    """Pins fail-closed: an unanswered question never turns into an approval."""

    async def run() -> tuple[ApprovalAnswer, str | None]:
        broker = ApprovalBroker(timeout_seconds=0.01)
        answer = await broker.decide("TEST_QUESTION")
        return answer, broker.pending_message()

    assert asyncio.run(run()) == ("cancel", None)


def test_given_questions_in_every_state_when_counted_then_only_waiting_ones_are_counted() -> None:
    """Pins that answered, timed-out and abandoned questions are removed, not left behind."""

    async def run() -> list[int]:
        broker = ApprovalBroker()
        counts = [broker.waiting_count()]
        first = asyncio.create_task(broker.decide("TEST_FIRST"))
        second = asyncio.create_task(broker.decide("TEST_SECOND"))
        await _let_tasks_run()
        counts.append(broker.waiting_count())
        broker.respond(ApprovalChoice.ACCEPT)
        await first
        counts.append(broker.waiting_count())
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        counts.append(broker.waiting_count())
        timed_out = ApprovalBroker(timeout_seconds=0.01)
        await timed_out.decide("TEST_SLOW")
        counts.append(timed_out.waiting_count())
        return counts

    assert asyncio.run(run()) == [0, 2, 1, 0, 0]


def test_given_a_cancelled_call_when_it_stops_waiting_then_its_question_is_no_longer_pending() -> (
    None
):
    """Pins that an abandoned question cannot be answered later by a stray click."""

    async def run() -> tuple[str | None, bool]:
        broker = ApprovalBroker()
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return broker.pending_message(), broker.respond(ApprovalChoice.ACCEPT)

    assert asyncio.run(run()) == (None, False)
