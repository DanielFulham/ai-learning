from pathlib import Path

import pytest

from infra.json_policy_store import JsonPolicyStore
from interfaces.policy_store_interface import InvalidPolicyError


def _make_store(tmp_path: Path, text: str | None = None) -> JsonPolicyStore:
    path = tmp_path / "state" / "permissions.json"
    if text is not None:
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
    return JsonPolicyStore(path)


def test_given_a_missing_file_when_loaded_then_no_rules_are_returned(tmp_path: Path) -> None:
    assert _make_store(tmp_path).load_rules() == {}


def test_given_a_missing_file_when_loaded_then_the_file_is_not_created(tmp_path: Path) -> None:
    """Pins that the store only reads: nothing the server does can write the policy."""
    _make_store(tmp_path).load_rules()
    assert not (tmp_path / "state").exists()


def test_given_a_valid_file_when_loaded_then_the_rules_come_back_as_written(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path, '{"TEST_TOOL_A": "allow", "TEST_TOOL_B": "alow"}')
    assert store.load_rules() == {"TEST_TOOL_A": "allow", "TEST_TOOL_B": "alow"}


def test_given_a_file_saved_with_a_byte_order_mark_when_loaded_then_it_still_parses(
    tmp_path: Path,
) -> None:
    """Windows editors add a BOM; a hand-edited policy file must not break on it."""
    path = tmp_path / "permissions.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"TEST_TOOL": "deny"}')
    assert JsonPolicyStore(path).load_rules() == {"TEST_TOOL": "deny"}


def test_given_a_file_saved_as_utf16_when_loaded_then_invalid_policy_is_raised(
    tmp_path: Path,
) -> None:
    """Pins F-L47-12: PowerShell's `echo >` writes UTF-16; that must fail closed, not crash."""
    path = tmp_path / "permissions.json"
    path.write_text('{"TEST_TOOL": "allow"}', encoding="utf-16")
    with pytest.raises(InvalidPolicyError):
        JsonPolicyStore(path).load_rules()


@pytest.mark.parametrize(
    "text",
    ["", "not json", "{", '["allow"]', '"allow"', "null", "42"],
)
def test_given_a_file_that_is_not_a_json_object_when_loaded_then_invalid_policy_is_raised(
    tmp_path: Path, text: str
) -> None:
    """Pins F-L47-5: the lab raised a bare JSON error at construction; this one fails closed."""
    with pytest.raises(InvalidPolicyError):
        _make_store(tmp_path, text).load_rules()


@pytest.mark.parametrize("text", ['{"TEST_TOOL": true}', '{"TEST_TOOL": 1}', '{"TEST_TOOL": null}'])
def test_given_a_rule_whose_value_is_not_text_when_loaded_then_invalid_policy_is_raised(
    tmp_path: Path, text: str
) -> None:
    with pytest.raises(InvalidPolicyError):
        _make_store(tmp_path, text).load_rules()
