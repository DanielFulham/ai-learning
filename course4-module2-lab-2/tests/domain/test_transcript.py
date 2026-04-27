from domain.transcript import _get_video_id, process, chunk_transcript

# --- _get_video_id ---

def test_get_video_id_valid_url():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert _get_video_id(url) == "dQw4w9WgXcQ"

def test_get_video_id_invalid_url():
    url = "https://www.notyoutube.com/watch?v=dQw4w9WgXcQ"
    assert _get_video_id(url) is None

# --- process ---

def test_process_object_format():
    class Snippet:
        def __init__(self, text, start):
            self.text = text
            self.start = start

    transcript = [Snippet("Hello world", 0), Snippet("This is a test", 5)]
    result = process(transcript)
    assert "Text: Hello world Start: 0" in result
    assert "Text: This is a test Start: 5" in result

def test_process_dict_format():
    transcript = [{"text": "Hello world", "start": 0.0}]
    result = process(transcript)
    assert "Hello world" in result

def test_process_empty_transcript():
    result = process([])
    assert result == ""

# --- chunk_transcript ---

def test_chunk_transcript_returns_list():
    text = "This is a test transcript. " * 20
    result = chunk_transcript(text)
    assert isinstance(result, list)

def test_chunk_transcript_respects_chunk_size():
    text = "This is a test transcript. " * 20
    result = chunk_transcript(text, chunk_size=50)
    assert all(len(chunk) <= 50 + 20 for chunk in result)  # +20 for overlap tolerance

def test_chunk_transcript_empty_string():
    result = chunk_transcript("")
    assert result == []