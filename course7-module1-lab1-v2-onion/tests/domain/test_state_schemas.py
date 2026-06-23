from typing import get_type_hints

from domain.auth_credentials import AuthCredentials
from domain.counter_tick import CounterTick
from domain.qa_exchange import QAExchange
from domain.state_schemas import AuthState, CounterState, QAState


def test_auth_state_declares_single_credentials_field_typed_as_auth_credentials() -> None:
    assert get_type_hints(AuthState) == {"credentials": AuthCredentials}


def test_qa_state_declares_single_exchange_field_typed_as_qa_exchange() -> None:
    assert get_type_hints(QAState) == {"exchange": QAExchange}


def test_counter_state_declares_single_tick_field_typed_as_counter_tick() -> None:
    assert get_type_hints(CounterState) == {"tick": CounterTick}