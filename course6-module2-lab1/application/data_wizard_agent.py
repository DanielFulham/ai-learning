from application.interfaces.data_wizard_agent_interface import DataWizardAgentInterface
from interfaces.tool_calling_agent_interface import ToolCallingAgentInterface


class DataWizardAgent(DataWizardAgentInterface):
    """Orchestrates data analysis queries via a tool-calling agent.

    Currently a thin facade over the agent. Exists as the named application service
    where orchestration concerns (logging, retries, pre/post processing) will land
    if needed. Mirrors the role of ChatService in Hooperman V2.
    """

    def __init__(self, agent: ToolCallingAgentInterface) -> None:
        self._agent = agent

    def ask(self, query: str) -> str:
        return self._agent.ask(query)