import pytest

from infra.console_user_input import ConsoleUserInput


def test_ask_passes_the_prompt_and_strips_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "  TEST_REPLY  "

    monkeypatch.setattr("builtins.input", fake_input)
    assert ConsoleUserInput().ask("TEST_PROMPT: ") == "TEST_REPLY"
    assert prompts == ["TEST_PROMPT: "]


def test_show_prints_the_text(capsys: pytest.CaptureFixture[str]) -> None:
    ConsoleUserInput().show("TEST_TEXT")
    assert capsys.readouterr().out == "TEST_TEXT\n"
