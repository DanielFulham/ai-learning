from infra.console_stream_consumer import ConsoleStreamConsumer


def test_on_update_prints_node_name_and_state_delta(capsys) -> None:
    consumer = ConsoleStreamConsumer()
    consumer.on_update("ValidateCredential", {"is_authenticated": True})
    captured = capsys.readouterr()
    assert "[ValidateCredential]" in captured.out
    assert "{'is_authenticated': True}" in captured.out


def test_on_update_handles_empty_state_delta(capsys) -> None:
    consumer = ConsoleStreamConsumer()
    consumer.on_update("SomeNode", {})
    captured = capsys.readouterr()
    assert "[SomeNode] {}" in captured.out


def test_on_update_returns_none(capsys) -> None:
    """Observational only — no return value to interfere with stream consumption."""
    consumer = ConsoleStreamConsumer()
    result = consumer.on_update("AnyNode", {"k": "v"})
    assert result is None