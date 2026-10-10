from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import download_samples


@pytest.fixture
def documents_dir(tmp_path: Path) -> Iterator[Path]:
    directory = tmp_path / "docs"
    with patch.object(download_samples, "DOCUMENTS_DIR", directory):
        yield directory


def fake_response(content: bytes) -> MagicMock:
    response = MagicMock()
    response.read.return_value = content
    response.__enter__.return_value = response
    return response


def test_download_writes_the_file(documents_dir: Path) -> None:
    documents_dir.mkdir()
    with patch("scripts.download_samples.urllib.request.urlopen") as urlopen:
        urlopen.return_value = fake_response(b"TEST_CONTENT")
        download_samples.download_sample_file("TEST_FILE.txt", "TEST_URL")
    assert (documents_dir / "TEST_FILE.txt").read_bytes() == b"TEST_CONTENT"


def test_download_skips_a_file_that_already_has_content(documents_dir: Path) -> None:
    documents_dir.mkdir()
    (documents_dir / "TEST_FILE.txt").write_bytes(b"TEST_EXISTING")
    with patch("scripts.download_samples.urllib.request.urlopen") as urlopen:
        download_samples.download_sample_file("TEST_FILE.txt", "TEST_URL")
        urlopen.assert_not_called()
    assert (documents_dir / "TEST_FILE.txt").read_bytes() == b"TEST_EXISTING"


def test_download_replaces_an_empty_file(documents_dir: Path) -> None:
    documents_dir.mkdir()
    (documents_dir / "TEST_FILE.txt").write_bytes(b"")
    with patch("scripts.download_samples.urllib.request.urlopen") as urlopen:
        urlopen.return_value = fake_response(b"TEST_CONTENT")
        download_samples.download_sample_file("TEST_FILE.txt", "TEST_URL")
    assert (documents_dir / "TEST_FILE.txt").read_bytes() == b"TEST_CONTENT"


def test_prepare_creates_the_directory_and_fetches_every_sample(documents_dir: Path) -> None:
    urls = {"TEST_A.txt": "TEST_URL_A", "TEST_B.txt": "TEST_URL_B"}
    with (
        patch.object(download_samples, "SAMPLE_FILE_URLS", urls),
        patch("scripts.download_samples.urllib.request.urlopen") as urlopen,
    ):
        urlopen.return_value = fake_response(b"TEST_CONTENT")
        download_samples.prepare_sample_files()
    assert sorted(path.name for path in documents_dir.iterdir()) == ["TEST_A.txt", "TEST_B.txt"]
