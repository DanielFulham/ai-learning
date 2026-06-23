from typing import Protocol

from domain.counter_tick import CounterTick


class CounterAgentServiceInterface(Protocol):
    """Service contract for the Counter workflow."""

    def run(self) -> CounterTick: ...