from typing import Protocol


class DataWizardAgentInterface(Protocol):
    """Contract consumed by the entry point.

    Defines what the entry point needs from the application service —
    not what any tool-calling agent provides. Implemented by DataWizardAgent.
    """

    def ask(self, query: str) -> str:
        """Run the agent and return the final answer."""
        ...