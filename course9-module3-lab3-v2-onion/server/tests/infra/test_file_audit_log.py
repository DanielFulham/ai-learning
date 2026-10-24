import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from infra.file_audit_log import FileAuditLog

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _make_log(tmp_path: Path) -> FileAuditLog:
    return FileAuditLog(tmp_path / "state" / "audit.log", clock=lambda: FIXED_TIME)


def _lines(tmp_path: Path) -> list[str]:
    return (tmp_path / "state" / "audit.log").read_text(encoding="utf-8").splitlines()


def test_given_a_new_log_when_an_entry_is_appended_then_the_file_has_one_line(
    tmp_path: Path,
) -> None:
    _make_log(tmp_path).append("TEST_TOOL", "{}", "allowed", "TEST_DETAIL")
    assert len(_lines(tmp_path)) == 1


def test_given_an_entry_when_appended_then_the_line_is_json_with_all_five_fields(
    tmp_path: Path,
) -> None:
    _make_log(tmp_path).append("TEST_TOOL", '{"filepath": "a.txt"}', "allowed", "TEST_DETAIL")
    assert json.loads(_lines(tmp_path)[0]) == {
        "time": "2026-01-02T03:04:05+00:00",
        "tool": "TEST_TOOL",
        "arguments": '{"filepath": "a.txt"}',
        "event": "allowed",
        "detail": "TEST_DETAIL",
    }


def test_given_arguments_when_appended_then_the_arguments_are_in_the_log(tmp_path: Path) -> None:
    """Pins F-L47-7: the lab's lines had no arguments, so nothing recorded what was run."""
    _make_log(tmp_path).append("write_file", '{"filepath": "TEST.txt"}', "succeeded", "")
    assert "TEST.txt" in _lines(tmp_path)[0]


def test_given_two_entries_when_appended_then_they_are_in_order_one_per_line(
    tmp_path: Path,
) -> None:
    log = _make_log(tmp_path)
    log.append("TEST_TOOL", "{}", "approval_requested", "")
    log.append("TEST_TOOL", "{}", "approved", "")
    events = [json.loads(line)["event"] for line in _lines(tmp_path)]
    assert events == ["approval_requested", "approved"]


@pytest.mark.parametrize("field", ["tool", "arguments_json", "event", "detail"])
def test_given_a_field_with_a_newline_and_a_fake_entry_when_appended_then_it_stays_one_line(
    tmp_path: Path, field: str
) -> None:
    """Pins F-L47-8: the lab wrote the command raw, so a newline could add a forged line."""
    forged = 'x\n{"time": "2026-01-01T00:00:00+00:00", "event": "denied"}'
    values = {"tool": "TEST_TOOL", "arguments_json": "{}", "event": "allowed", "detail": ""}
    values[field] = forged
    _make_log(tmp_path).append(
        values["tool"], values["arguments_json"], values["event"], values["detail"]
    )
    lines = _lines(tmp_path)
    assert len(lines) == 1
    assert forged in json.loads(lines[0]).values()


def test_given_a_missing_folder_when_appended_then_the_folder_is_created(tmp_path: Path) -> None:
    _make_log(tmp_path).append("TEST_TOOL", "{}", "allowed", "")
    assert (tmp_path / "state").is_dir()


def test_given_an_existing_log_when_a_new_log_object_appends_then_earlier_lines_stay(
    tmp_path: Path,
) -> None:
    """Pins append-only behaviour across restarts: a new process never truncates the trail."""
    _make_log(tmp_path).append("TEST_TOOL", "{}", "allowed", "first")
    _make_log(tmp_path).append("TEST_TOOL", "{}", "allowed", "second")
    assert [json.loads(line)["detail"] for line in _lines(tmp_path)] == ["first", "second"]


def test_given_non_ascii_text_when_appended_then_it_round_trips(tmp_path: Path) -> None:
    _make_log(tmp_path).append("TEST_TOOL", "{}", "allowed", "TEST_é✓")
    assert json.loads(_lines(tmp_path)[0])["detail"] == "TEST_é✓"


def test_given_a_missing_log_when_read_then_the_text_is_empty(tmp_path: Path) -> None:
    assert _make_log(tmp_path).read_text() == ""


def test_given_entries_when_read_then_the_text_is_the_file_content(tmp_path: Path) -> None:
    log = _make_log(tmp_path)
    log.append("TEST_TOOL", "{}", "allowed", "first")
    log.append("TEST_TOOL", "{}", "allowed", "second")
    assert log.read_text() == (tmp_path / "state" / "audit.log").read_text(encoding="utf-8")


def test_given_no_clock_when_appended_then_the_time_is_timezone_aware_utc(
    tmp_path: Path,
) -> None:
    log = FileAuditLog(tmp_path / "audit.log")
    log.append("TEST_TOOL", "{}", "allowed", "")
    line = (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()[0]
    stamp = datetime.fromisoformat(json.loads(line)["time"])
    assert stamp.utcoffset() == UTC.utcoffset(None)
