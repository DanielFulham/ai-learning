from datetime import datetime
from typing import Any, Callable
from uuid import UUID, uuid4

from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    ModelInvocationFailed,
    QAEvent,
)
from domain.qa_exchange import QAExchange


_CONTEXT_NODE_NAME = "ContextNode"
_QA_NODE_NAME = "QANode"


def translate_qa_update(
    node_name: str,
    state_delta: dict[str, Any],
    run_id: UUID,
    clock: Callable[[], datetime],
) -> list[QAEvent]:
    """Translate one QA node update into the events it represents.

    Pure function: same inputs produce the same outputs except for event_id
    (uuid4 at translation time) and occurred_at (clock at translation
    time). Both timing concerns are explicit — event_id is generated here
    by design (translator is the canonical source per the V3a pin),
    occurred_at is injected for deterministic tests and pluggable
    production clocks.

    The return type is `list[QAEvent]` even though V3a's translator always
    emits exactly one event per update. The list shape leaves room for a
    future update to fan out into multiple events without breaking the
    type contract.

    Two-stage narrowing applies at the framework boundary: the caller
    has already isinstance-checked the chunk against `dict` to access
    `state_delta`; this function performs the second narrow against the
    domain type (`QAExchange`).

    Unknown node names raise. The QA workflow is enumerated and small;
    a surprising node name is a signal that the graph changed without
    the translator being updated, and silent passthrough hides that.

    `QuestionReceived` is NOT emitted here. Run-lifecycle events are the
    agent service's responsibility, fired before `graph.stream()` starts.
    The translator handles node updates only.
    """
    exchange = state_delta.get("exchange")
    if not isinstance(exchange, QAExchange):
        raise ValueError(
            f"QA state delta from node '{node_name}' missing or malformed "
            f"'exchange' field; expected QAExchange, got "
            f"{type(exchange).__name__}"
        )

    if node_name == _CONTEXT_NODE_NAME:
        return [
            ContextRetrieved(
                event_id=uuid4(),
                aggregate_id=run_id,
                occurred_at=clock(),
                context=exchange.context,
            )
        ]

    if node_name == _QA_NODE_NAME:
        if exchange.error_info is not None:
            # error_info present → failure path; answer (if present) is the
            # user-safe message, not a successful generation
            return [
                ModelInvocationFailed(
                    event_id=uuid4(),
                    aggregate_id=run_id,
                    occurred_at=clock(),
                    error_info=exchange.error_info,
                )
            ]
        if exchange.answer is None:
            raise ValueError(
                f"QANode update has neither answer nor error_info on the "
                f"exchange — QANode must populate one of them before "
                f"yielding the update"
            )
        return [
            AnswerGenerated(
                event_id=uuid4(),
                aggregate_id=run_id,
                occurred_at=clock(),
                answer=exchange.answer,
            )
        ]

    raise ValueError(
        f"Unknown QA node '{node_name}' — translator handles "
        f"'{_CONTEXT_NODE_NAME}' and '{_QA_NODE_NAME}' only"
    )