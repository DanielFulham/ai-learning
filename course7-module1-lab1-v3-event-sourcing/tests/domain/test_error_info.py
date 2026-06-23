from dataclasses import FrozenInstanceError

import pytest

from domain.error_info import ErrorInfo


def _make_error_info(
    exception_type: str = "OllamaConnectionError",
    exception_message: str = "Connection refused",
) -> ErrorInfo:
    return ErrorInfo(
        exception_type=exception_type,
        exception_message=exception_message,
    )


class TestErrorInfo:

    def test_fields_accessible(self) -> None:
        info = _make_error_info(
            exception_type="httpx.HTTPError",
            exception_message="503 Service Unavailable",
        )
        assert info.exception_type == "httpx.HTTPError"
        assert info.exception_message == "503 Service Unavailable"

    def test_is_frozen(self) -> None:
        """Pinned: ErrorInfo is immutable. Mutation would let storage and
        translator paths drift apart after construction."""
        info = _make_error_info()
        with pytest.raises(FrozenInstanceError):
            setattr(info, "exception_message", "different")

    def test_equality_by_value(self) -> None:
        """Pinned: frozen dataclass equality is structural, not identity.
        Two ErrorInfo instances with the same fields compare equal — lets
        tests assert event payload contents without holding references."""
        assert _make_error_info() == _make_error_info()
        assert _make_error_info(exception_type="A") != _make_error_info(exception_type="B")