import os
import sys
from pathlib import Path

import pytest

from infra.local_file_store import LocalFileStore
from interfaces.file_store_interface import PathOutsideRootError


def _make_store(tmp_path: Path) -> LocalFileStore:
    return LocalFileStore(tmp_path / "root")


def _outside_paths(tmp_path: Path) -> list[str]:
    return ["../outside.txt", "sub/../../outside.txt", str(tmp_path / "outside.txt")]


def test_root_folder_is_created(tmp_path: Path) -> None:
    _make_store(tmp_path)
    assert (tmp_path / "root").is_dir()


def test_a_relative_root_is_resolved_when_the_store_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The check compares against the resolved root, so it must not move with the working directory.
    monkeypatch.chdir(tmp_path)
    store = LocalFileStore(Path("rel_root"))
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    store.write_text("a.txt", "TEST_CONTENT")
    assert (tmp_path / "rel_root" / "a.txt").read_text(encoding="utf-8") == "TEST_CONTENT"


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    assert store.read_text("a.txt") == "TEST_CONTENT"


def test_write_creates_parent_folders(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("sub/deeper/a.txt", "TEST_CONTENT")
    assert store.kind_of("sub/deeper") == "directory"


def test_write_replaces_existing_content(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "one")
    store.write_text("a.txt", "two")
    assert store.read_text("a.txt") == "two"


def test_write_and_read_use_utf8(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_é✓")
    assert (tmp_path / "root" / "a.txt").read_bytes() == "TEST_é✓".encode()
    assert store.read_text("a.txt") == "TEST_é✓"


def test_kind_of_reports_file_directory_and_missing(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("sub/a.txt", "TEST_CONTENT")
    assert store.kind_of("sub/a.txt") == "file"
    assert store.kind_of("sub") == "directory"
    assert store.kind_of("nope.txt") == "missing"


def test_kind_of_the_root_is_a_directory(tmp_path: Path) -> None:
    assert _make_store(tmp_path).kind_of(".") == "directory"


def test_read_of_a_missing_file_raises(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read_text("nope.txt")


def test_list_entries_is_sorted_with_names_sizes_and_kinds(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("b.txt", "12345")
    store.write_text("a.txt", "1")
    store.write_text("sub/c.txt", "TEST_CONTENT")
    entries = store.list_entries(".")
    assert [e["name"] for e in entries] == ["a.txt", "b.txt", "sub"]
    assert [e["is_dir"] for e in entries] == [False, False, True]
    assert [e["size"] for e in entries] == [1, 5, 0]


def test_list_entries_gives_paths_relative_to_the_root(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "1")
    store.write_text("sub/c.txt", "TEST_CONTENT")
    assert [e["path"] for e in store.list_entries(".")] == ["a.txt", "sub"]
    assert [e["path"] for e in store.list_entries("sub")] == [str(Path("sub") / "c.txt")]


def test_list_entries_of_an_empty_folder_is_empty(tmp_path: Path) -> None:
    assert _make_store(tmp_path).list_entries(".") == []


def test_path_that_goes_up_and_back_inside_is_allowed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.write_text("a.txt", "TEST_CONTENT")
    assert store.read_text("sub/../a.txt") == "TEST_CONTENT"


@pytest.mark.parametrize("which", range(3))
def test_every_method_refuses_a_path_outside_the_root(tmp_path: Path, which: int) -> None:
    store = _make_store(tmp_path)
    bad = _outside_paths(tmp_path)[which]
    with pytest.raises(PathOutsideRootError):
        store.kind_of(bad)
    with pytest.raises(PathOutsideRootError):
        store.read_text(bad)
    with pytest.raises(PathOutsideRootError):
        store.write_text(bad, "TEST_CONTENT")
    with pytest.raises(PathOutsideRootError):
        store.list_entries(bad)


def test_listing_the_parent_of_the_root_is_refused(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(PathOutsideRootError):
        store.list_entries("..")


def test_refused_write_leaves_nothing_outside_the_root(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(PathOutsideRootError):
        store.write_text("../outside.txt", "TEST_CONTENT")
    assert not (tmp_path / "outside.txt").exists()


def test_refused_read_does_not_return_the_outside_file(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    (tmp_path / "outside.txt").write_text("TEST_SECRET", encoding="utf-8")
    with pytest.raises(PathOutsideRootError):
        store.read_text("../outside.txt")


@pytest.mark.skipif(sys.platform != "win32", reason="backslash is a separator only on Windows")
def test_backslash_traversal_is_refused_on_windows(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(PathOutsideRootError):
        store.read_text("..\\outside.txt")


def test_a_link_that_points_outside_the_root_is_refused(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    (tmp_path / "outside.txt").write_text("TEST_SECRET", encoding="utf-8")
    try:
        os.symlink(tmp_path / "outside.txt", tmp_path / "root" / "link.txt")
    except OSError:
        pytest.skip("this machine does not allow creating symbolic links")
    with pytest.raises(PathOutsideRootError):
        store.read_text("link.txt")
