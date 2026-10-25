from application.mappers import parse_policy_rules
from domain.default_policies import DEFAULT_POLICIES
from domain.policy import Policy
from interfaces.policy_store_interface import PolicyStoreInterface


class PolicyBook:
    """Answers "what is the policy for this tool?" from the stored rules plus the defaults.

    The store is read on every question, so an edit to the policy file applies to the next
    call. One bad value makes the whole file invalid: InvalidPolicyError is raised for any tool.
    """

    def __init__(self, store: PolicyStoreInterface) -> None:
        self._store = store

    def all_policies(self) -> dict[str, Policy]:
        return {**DEFAULT_POLICIES, **parse_policy_rules(self._store.load_rules())}

    def policy_for(self, tool: str) -> Policy:
        return self.all_policies().get(tool, Policy.ASK)
