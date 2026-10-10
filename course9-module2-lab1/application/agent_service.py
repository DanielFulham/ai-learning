from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from application.mappers import to_agent_reply
from domain.conversation import AgentReply
from interfaces.tool_source_interface import ToolSourceInterface


class AgentService:
    """Runs a tool-using agent against whichever tool source it was given."""

    def __init__(self, model: BaseChatModel, tool_source: ToolSourceInterface) -> None:
        self._model = model
        self._tool_source = tool_source

    async def ask(self, prompt: str) -> AgentReply:
        # The agent has to run while the tool source is open: MCP tools use its live session.
        async with self._tool_source.open() as tools:
            agent = create_agent(model=self._model, tools=list(tools))
            result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
        return to_agent_reply(result["messages"])
