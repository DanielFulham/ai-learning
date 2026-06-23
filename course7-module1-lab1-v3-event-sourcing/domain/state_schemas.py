from typing import TypedDict

from domain.qa_exchange import QAExchange


class QAState(TypedDict, total=False):
    """LangGraph state schema for the QA workflow.

    Carries one domain object — the V2 'state as a single domain object
    field' pattern. The TypedDict declares exactly one key; nodes use
    `dataclasses.replace()` to update the QAExchange and return
    `{"exchange": new_exchange}`. LangGraph's last-write-wins merge keeps
    the contract intact.

    `AuthState` and `CounterState` land here when V3b and V3c lift their
    respective workflows.
    """

    exchange: QAExchange