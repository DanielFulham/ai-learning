from infra.null_stream_consumer import NullStreamConsumer
from interfaces.stream_consumer_interface import StreamConsumerInterface


def _accepts_consumer(consumer: StreamConsumerInterface) -> None:
    """Type-guard helper. Pyright fails the call site if structural
    typing breaks."""


class TestNullStreamConsumer:

    def test_satisfies_stream_consumer_interface(self) -> None:
        _accepts_consumer(NullStreamConsumer())

    def test_on_update_returns_none(self) -> None:
        """Pinned: NullStreamConsumer is a no-op. on_update completes
        without raising and returns None."""
        consumer = NullStreamConsumer()
        result = consumer.on_update("AnyNode", {"any": "delta"})
        assert result is None