import os
import sys
from pathlib import Path

import pytest

from infra.local_file_store import LocalFileStore
from interfaces.file_store_interface import PathOutsideRootError

OPERATIONS = ["exists", "read_text", "write_text", "delete"]
OUTSIDE_KINDS = ["parent", "nested_parent", "absolute"]


def _make_store(tmp_path: Path) -> LocalFileStore:
    return LocalFileStore(tmp_path / "root")


def _outside_path(tmp_path: Path, kind: str) -> str:
    if kind == "parent":
        return "../outside.txt"
    if kind == "nested_parent":
        return "sub/../../outside.txt"
    return str(tmp_path / "outside.txt")


def _call(store: LocalFileStore, operation: str, path: str) -> object:
    if operation == "exists":
        return store.exists(path)
    if operation == "read_text":
        return store.read_text(path)
    if operation == "write_text":
        return store.write_text(path, "TEST_CONTENT")
    return store.delete(path)


def test_given_a_new_store_when_built_then_the_root_folder_exists(tmp_path: Path) -> None:
    _make_store(tmp_path)
    assert (tmp_path / "root").is_dir()


def test_given_a_relative_root_when_the_working_directory_changes_then_it_stays_put(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins that the root is resolved at construction, so the check cannot drift with the cwd."""
    monkeypatch.chdir(tmp_path)
    store = LocalFileStore(Path("rel_root"))
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    store.write_text("a.txt", "TEST_CONTENT")
    assert (tmp_path / "rel_root" / "a.txt").read_text(encoding="utf-8") == "TEST_CONTENT"


def test_given_written_text_when_read_back_then_it_is_the_same_text(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    assert store.read_text("a.txt") == "TEST_CONTENT"


def test_given_a_nested_path_when_written_then_the_parent_folders_are_created(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    store.write_text("sub/deeper/a.txt", "TEST_CONTENT")
    assert (tmp_path / "root" / "sub" / "deeper" / "a.txt").is_file()


def test_given_an_existing_file_when_written_again_then_the_content_is_replaced(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "one")
    store.write_text("a.txt", "two")
    assert store.read_text("a.txt") == "two"


def test_given_non_ascii_text_when_written_and_read_then_it_is_utf8_on_disk(
    tmp_path: Path,
) -> None:
    """Pins F-L47-12: the lab read and wrote with the platform default (cp1252 here)."""
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_é✓")
    assert (tmp_path / "root" / "a.txt").read_bytes() == "TEST_é✓".encode()
    assert store.read_text("a.txt") == "TEST_é✓"


def test_given_a_file_that_is_not_there_when_read_then_file_not_found_is_raised(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _make_store(tmp_path).read_text("nope.txt")


def test_given_a_file_and_a_missing_name_when_asked_if_they_exist_then_true_and_false(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    assert store.exists("a.txt") is True
    assert store.exists("nope.txt") is False


def test_given_a_folder_when_asked_if_it_exists_then_false_because_only_files_count(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    store.write_text("sub/a.txt", "TEST_CONTENT")
    assert store.exists("sub") is False


def test_given_a_file_when_deleted_then_it_is_gone(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    store.delete("a.txt")
    assert not (tmp_path / "root" / "a.txt").exists()


def test_given_a_missing_file_when_deleted_then_file_not_found_is_raised(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _make_store(tmp_path).delete("nope.txt")


def test_given_a_missing_file_when_read_then_the_message_names_the_relative_path_only(
    tmp_path: Path,
) -> None:
    """Pins that errors never leak the server's folder layout to a client or a model."""
    with pytest.raises(FileNotFoundError) as error:
        _make_store(tmp_path).read_text("TEST_MISSING.txt")
    assert "TEST_MISSING.txt" in str(error.value)
    # Windows doubles the backslashes inside an OSError message, so check a folder name instead.
    assert tmp_path.name not in str(error.value)


def test_given_a_missing_file_when_deleted_then_the_message_names_the_relative_path_only(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError) as error:
        _make_store(tmp_path).delete("TEST_MISSING.txt")
    assert "TEST_MISSING.txt" in str(error.value)
    assert tmp_path.name not in str(error.value)


def test_given_a_path_under_a_file_when_written_then_the_message_hides_the_server_folder(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    with pytest.raises(OSError) as error:
        store.write_text("a.txt/b.txt", "TEST_CONTENT")
    assert "a.txt/b.txt" in str(error.value)
    assert tmp_path.name not in str(error.value)


@pytest.mark.parametrize("operation", ["read_text", "write_text", "delete"])
def test_given_the_root_or_a_folder_when_used_as_a_file_then_is_a_directory_is_raised(
    tmp_path: Path, operation: str
) -> None:
    """Pins that a delete can never remove the root or a folder, only a single file."""
    store = _make_store(tmp_path)
    store.write_text("sub/a.txt", "TEST_CONTENT")
    for path in (".", "", "sub"):
        with pytest.raises(IsADirectoryError):
            _call(store, operation, path)
    assert (tmp_path / "root" / "sub" / "a.txt").is_file()


def test_given_a_path_that_goes_up_and_back_inside_when_read_then_it_is_allowed(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    assert store.read_text("sub/../a.txt") == "TEST_CONTENT"


@pytest.mark.parametrize("kind", OUTSIDE_KINDS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_given_a_path_outside_the_root_when_any_operation_runs_then_it_is_refused(
    tmp_path: Path, operation: str, kind: str
) -> None:
    """Pins F-L47-6: the lab's read_file returned ../ and absolute paths with no check."""
    store = _make_store(tmp_path)
    with pytest.raises(PathOutsideRootError):
        _call(store, operation, _outside_path(tmp_path, kind))


def test_given_a_refused_write_when_it_fails_then_nothing_is_written_outside(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(PathOutsideRootError):
        store.write_text("../outside.txt", "TEST_CONTENT")
    assert not (tmp_path / "outside.txt").exists()


def test_given_a_refused_delete_when_it_fails_then_the_outside_file_is_untouched(
    tmp_path: Path,
) -> None:
    (tmp_path / "outside.txt").write_text("TEST_SECRET", encoding="utf-8")
    with pytest.raises(PathOutsideRootError):
        _make_store(tmp_path).delete("../outside.txt")
    assert (tmp_path / "outside.txt").read_text(encoding="utf-8") == "TEST_SECRET"


def test_given_a_refused_read_when_it_fails_then_the_outside_text_is_not_returned(
    tmp_path: Path,
) -> None:
    (tmp_path / "outside.txt").write_text("TEST_SECRET", encoding="utf-8")
    with pytest.raises(PathOutsideRootError):
        _make_store(tmp_path).read_text("../outside.txt")


def test_given_a_name_with_an_encoded_slash_when_read_then_it_is_just_a_missing_name(
    tmp_path: Path,
) -> None:
    """Pins that %2F is not decoded here: the form that got through in Lab 45 is a plain name."""
    (tmp_path / "outside.txt").write_text("TEST_SECRET", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        _make_store(tmp_path).read_text("..%2Foutside.txt")


@pytest.mark.skipif(sys.platform != "win32", reason="backslash is a separator only on Windows")
def test_given_backslash_traversal_when_read_on_windows_then_it_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(PathOutsideRootError):
        _make_store(tmp_path).read_text("..\\outside.txt")


def test_given_a_link_pointing_outside_the_root_when_read_then_it_is_refused(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    (tmp_path / "outside.txt").write_text("TEST_SECRET", encoding="utf-8")
    try:
        os.symlink(tmp_path / "outside.txt", tmp_path / "root" / "link.txt")
    except OSError:
        pytest.skip("this machine does not allow creating symbolic links")
    with pytest.raises(PathOutsideRootError):
        store.read_text("link.txt")
