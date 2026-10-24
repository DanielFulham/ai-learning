from collections.abc import Mapping

from domain.policy import Policy
from interfaces.policy_store_interface import InvalidPolicyError


def parse_policy_rules(raw_rules: Mapping[str, str]) -> dict[str, Policy]:
    """Turn the stored tool -> text rules into tool -> Policy, refusing anything unclear."""
    rules: dict[str, Policy] = {}
    for tool, text in raw_rules.items():
        if ":" in tool:
            raise InvalidPolicyError(
                f"'{tool}' looks like an argument-level rule; only tool names are supported"
            )
        try:
            rules[tool] = Policy(text)
        except ValueError as error:
            raise InvalidPolicyError(
                f"The policy for '{tool}' must be allow, deny or ask, not '{text}'"
            ) from error
    return rules
