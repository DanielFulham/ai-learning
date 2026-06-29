from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from domain.auth_credentials import AuthCredentials
from domain.auth_result import AuthResult


class TestAuthResult:

    def test_fields_accessible(self) -> None:
        run_id = uuid4()
        credentials = AuthCredentials(username="alice", is_authenticated=True)
        result = AuthResult(credentials=credentials, run_id=run_id)
        assert result.credentials == credentials
        assert result.run_id == run_id

    def test_is_frozen(self) -> None:
        result = AuthResult(credentials=AuthCredentials(), run_id=uuid4())
        with pytest.raises(FrozenInstanceError):
            setattr(result, "run_id", uuid4())

    def test_structural_equality(self) -> None:
        """Pinned: equality compares both fields. Two AuthResults with the
        same credentials and same run_id compare equal — lets tests assert
        on result contents without holding references."""
        run_id = uuid4()
        credentials = AuthCredentials(username="alice", is_authenticated=True)
        a = AuthResult(credentials=credentials, run_id=run_id)
        b = AuthResult(credentials=credentials, run_id=run_id)
        assert a == b

    def test_different_run_id_not_equal(self) -> None:
        credentials = AuthCredentials(username="alice", is_authenticated=True)
        a = AuthResult(credentials=credentials, run_id=uuid4())
        b = AuthResult(credentials=credentials, run_id=uuid4())
        assert a != b
