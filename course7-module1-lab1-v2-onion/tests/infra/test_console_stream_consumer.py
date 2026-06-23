from infra.console_stream_consumer import ConsoleStreamConsumer
from domain.auth_credentials import AuthCredentials

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


def test_on_update_redacts_password_field_on_dataclasses(capsys) -> None:
    """Pin the tactical redaction — passwords on dataclass state deltas
    must not appear in the printed output. V3 replaces this with proper
    domain-declared sensitive-field policy."""
    consumer = ConsoleStreamConsumer()
    consumer.on_update(
        "InputNode",
        {"credentials": AuthCredentials(username="test_user", password="secret123")},
    )
    captured = capsys.readouterr()
    assert "secret123" not in captured.out
    assert "***" in captured.out
    assert "test_user" in captured.out