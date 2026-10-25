import asyncio
import threading
import time

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


def test_given_waiting_questions_when_all_are_cancelled_then_each_call_gets_cancel() -> None:
    """Pins F-L47-27: a closed page must release its waiting calls, never approve them."""

    async def run() -> tuple[list[ApprovalAnswer], int, str | None]:
        broker = ApprovalBroker()
        first = asyncio.create_task(broker.decide("TEST_FIRST"))
        second = asyncio.create_task(broker.decide("TEST_SECOND"))
        await _let_tasks_run()
        broker.cancel_all()
        answers: list[ApprovalAnswer] = [await first, await second]
        return answers, broker.waiting_count(), broker.pending_message()

    assert asyncio.run(run()) == (["cancel", "cancel"], 0, None)


def test_given_nothing_waiting_when_all_are_cancelled_then_nothing_breaks() -> None:
    ApprovalBroker().cancel_all()


def test_given_a_question_waiting_when_a_worker_thread_responds_then_the_call_wakes_at_once() -> (
    None
):
    """Pins F-L47-29: Gradio runs button callbacks on worker threads, not on the loop's thread."""

    async def run() -> tuple[ApprovalAnswer, float]:
        broker = ApprovalBroker(timeout_seconds=5)
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        started = time.monotonic()
        threading.Timer(0.05, broker.respond, args=(ApprovalChoice.ACCEPT,)).start()
        answer = await asyncio.wait_for(task, 2)
        return answer, time.monotonic() - started

    answer, waited = asyncio.run(run())
    assert answer == "accept"
    assert waited < 1


def test_given_one_question_when_two_threads_respond_then_only_one_answer_counts() -> None:
    """Pins F-L47-29: the second click must not also claim the same question."""

    async def run() -> list[bool]:
        broker = ApprovalBroker()
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        results: list[bool] = []
        worker = threading.Thread(
            target=lambda: results.extend(
                [broker.respond(ApprovalChoice.ACCEPT), broker.respond(ApprovalChoice.DECLINE)]
            )
        )
        worker.start()
        await asyncio.to_thread(worker.join)
        assert await asyncio.wait_for(task, 2) == "accept"
        return results

    assert asyncio.run(run()) == [True, False]


def test_given_a_waiting_question_when_a_thread_cancels_all_then_the_call_wakes_at_once() -> None:
    async def run() -> ApprovalAnswer:
        broker = ApprovalBroker(timeout_seconds=5)
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        threading.Timer(0.05, broker.cancel_all).start()
        return await asyncio.wait_for(task, 2)

    assert asyncio.run(run()) == "cancel"


def test_given_a_question_that_was_cancelled_when_the_user_responds_then_nothing_is_waiting() -> (
    None
):
    async def run() -> tuple[bool, str | None]:
        broker = ApprovalBroker()
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        broker.cancel_all()
        late = broker.respond(ApprovalChoice.ACCEPT)
        await task
        return late, broker.pending_message()

    assert asyncio.run(run()) == (False, None)


def test_given_a_question_just_answered_when_asked_for_the_pending_one_then_there_is_none() -> None:
    """Pins that a claimed question is hidden at once, before the waiting call has woken."""

    async def run() -> str | None:
        broker = ApprovalBroker()
        task = asyncio.create_task(broker.decide("TEST_QUESTION"))
        await _let_tasks_run()
        broker.respond(ApprovalChoice.ACCEPT)
        pending = broker.pending_message()
        await task
        return pending

    assert asyncio.run(run()) is None


def test_given_no_turn_running_when_one_begins_then_it_is_allowed() -> None:
    assert ApprovalBroker().begin_turn() is True


def test_given_a_turn_running_when_another_begins_then_it_is_refused() -> None:
    """Pins F-L47-32: one ordinary turn per session, so two cannot overwrite each other."""
    broker = ApprovalBroker()
    broker.begin_turn()
    assert broker.begin_turn() is False


def test_given_a_turn_that_ended_when_another_begins_then_it_is_allowed() -> None:
    broker = ApprovalBroker()
    broker.begin_turn()
    broker.end_turn()
    assert broker.begin_turn() is True


def test_given_two_threads_beginning_a_turn_at_once_when_both_try_then_only_one_wins() -> None:
    broker = ApprovalBroker()
    results: list[bool] = []
    barrier = threading.Barrier(8)

    def attempt() -> None:
        barrier.wait()
        results.append(broker.begin_turn())

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1
