"""Tests for storyteller.py."""

import sys
from unittest.mock import patch, MagicMock

import pytest

import storyteller


@patch("storyteller.gTTS")
@patch("storyteller.OllamaLLM")
def test_main_runs_full_pipeline(mock_ollama_cls, mock_gtts_cls, tmp_path, monkeypatch):
    """main() should: parse argv, build the model, generate a story, save MP3."""

    # Arrange — fake the LLM
    fake_story = "Once upon a time, there was a butterfly."
    mock_model = MagicMock()
    mock_model.invoke.return_value = fake_story
    mock_ollama_cls.return_value = mock_model

    # Arrange — fake gTTS
    mock_tts = MagicMock()
    mock_gtts_cls.return_value = mock_tts

    # Arrange — redirect output dir to a tmp folder so we don't litter
    monkeypatch.setattr(storyteller, "OUTPUT_DIR", tmp_path)

    # Arrange — simulate CLI args
    monkeypatch.setattr(sys, "argv", ["storyteller.py", "butterflies"])

    # Act
    storyteller.main()

    # Assert — model was instantiated with our config
    mock_ollama_cls.assert_called_once_with(
        model=storyteller.MODEL_ID,
        temperature=0.0,
        num_predict=1000,
    )

    # Assert — model.invoke was called with a prompt mentioning the topic
    mock_model.invoke.assert_called_once()
    prompt_arg = mock_model.invoke.call_args[0][0]
    assert "butterflies" in prompt_arg

    # Assert — gTTS got the model's output
    mock_gtts_cls.assert_called_once_with(fake_story)

    # Assert — save was called with a path inside our tmp dir
    mock_tts.save.assert_called_once()
    save_path = mock_tts.save.call_args[0][0]
    assert str(tmp_path) in save_path
    assert save_path.endswith(".mp3")


def test_main_uses_default_topic_when_no_argv(monkeypatch):
    """When no CLI arg is passed, main() should fall back to DEFAULT_TOPIC."""

    with patch("storyteller.OllamaLLM") as mock_ollama_cls, \
         patch("storyteller.gTTS") as mock_gtts_cls:

        mock_model = MagicMock()
        mock_model.invoke.return_value = "fake story"
        mock_ollama_cls.return_value = mock_model
        mock_gtts_cls.return_value = MagicMock()

        monkeypatch.setattr(sys, "argv", ["storyteller.py"])  # no topic arg

        storyteller.main()

        prompt_arg = mock_model.invoke.call_args[0][0]
        assert storyteller.DEFAULT_TOPIC in prompt_arg