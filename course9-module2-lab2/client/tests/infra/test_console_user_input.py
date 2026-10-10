import pytest

from infra.console_user_input import ConsoleUserInput


def test_ask_prompts_and_strips(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "  TEST_ANSWER  "

    monkeypatch.setattr("builtins.input", fake_input)
    assert ConsoleUserInput().ask("TEST_PROMPT") == "TEST_ANSWER"
    assert prompts == ["TEST_PROMPT"]


def test_show_prints(capsys: pytest.CaptureFixture[str]) -> None:
    ConsoleUserInput().show("TEST_TEXT")
    assert capsys.readouterr().out == "TEST_TEXT\n"
