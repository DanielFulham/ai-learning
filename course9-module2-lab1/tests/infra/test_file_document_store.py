from pathlib import Path

import pytest

from infra.file_document_store import FileDocumentStore


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A store root with one document, plus a secret file next to it, outside the root."""
    store_root = tmp_path / "docs"
    store_root.mkdir()
    (store_root / "TEST_DOC.txt").write_text("TEST_CONTENT", encoding="utf-8")
    (tmp_path / "TEST_SECRET.txt").write_text("TEST_SECRET_CONTENT", encoding="utf-8")
    return store_root


def test_reads_an_existing_document(root: Path) -> None:
    assert FileDocumentStore(root).read("TEST_DOC.txt") == "TEST_CONTENT"


def test_reads_utf8_content(root: Path) -> None:
    (root / "TEST_UTF8.txt").write_text("café ✓", encoding="utf-8")
    assert FileDocumentStore(root).read("TEST_UTF8.txt") == "café ✓"


def test_missing_document_returns_none(root: Path) -> None:
    assert FileDocumentStore(root).read("TEST_MISSING.txt") is None


def test_directory_name_returns_none(root: Path) -> None:
    (root / "TEST_DIR").mkdir()
    assert FileDocumentStore(root).read("TEST_DIR") is None


def test_empty_name_returns_none(root: Path) -> None:
    assert FileDocumentStore(root).read("") is None


@pytest.mark.parametrize("name", ["../TEST_SECRET.txt", "..\\TEST_SECRET.txt"])
def test_name_that_climbs_out_of_the_root_returns_none(root: Path, name: str) -> None:
    store = FileDocumentStore(root)
    assert store.read(name) is None


def test_absolute_path_outside_the_root_returns_none(root: Path) -> None:
    secret = root.parent / "TEST_SECRET.txt"
    assert FileDocumentStore(root).read(str(secret)) is None


def test_dotdot_that_stays_inside_the_root_is_allowed(root: Path) -> None:
    (root / "sub").mkdir()
    assert FileDocumentStore(root).read("sub/../TEST_DOC.txt") == "TEST_CONTENT"
