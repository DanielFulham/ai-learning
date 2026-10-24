import json
from pathlib import Path

from interfaces.policy_store_interface import InvalidPolicyError, PolicyStoreInterface


class JsonPolicyStore(PolicyStoreInterface):
    """Reads tool policies from a JSON file shaped like {"tool_name": "allow"}.

    Read-only: nothing here writes the file, so the server cannot change its own policy. A file
    that is missing means "no rules". A file that cannot be trusted raises InvalidPolicyError.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def load_rules(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            # utf-8-sig accepts the byte-order mark some Windows editors add.
            data = json.loads(self._path.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidPolicyError(
                f"The policy file {self._path.name} is not valid UTF-8 JSON: {error}"
            ) from error
        if not isinstance(data, dict):
            raise InvalidPolicyError(
                f"The policy file {self._path.name} must hold a JSON object of tool: policy"
            )
        rules: dict[str, str] = {}
        for tool, value in data.items():
            if not isinstance(value, str):
                raise InvalidPolicyError(
                    f"The policy for '{tool}' in {self._path.name} must be text, not {value!r}"
                )
            rules[str(tool)] = value
        return rules
