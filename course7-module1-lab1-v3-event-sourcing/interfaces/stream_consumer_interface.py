from typing import Any, Protocol


class StreamConsumerInterface(Protocol):
    """Receives node-update tuples from a LangGraph stream.

    Implementations choose what to do with each update — print, log,
    persist as events, do nothing. Service code is agnostic to the
    concrete; the seam supports multiple consumers behind one interface
    (the decorator pattern, used by `EventTranslatingStreamConsumer` to
    wrap an inner consumer and add event-store writes).
    """

    def on_update(self, node_name: str, state_delta: dict[str, Any]) -> None: ...