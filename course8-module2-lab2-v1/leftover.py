from typing import ClassVar

from crewai import LLM, Agent, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, task


@CrewBase
class LeftoversCrew:
    """CrewBase class loading the leftover_manager agent and leftover_task from YAML."""

    agents_config: ClassVar[str | dict[str, dict[str, str]]] = "config/agents.yaml"
    tasks_config: ClassVar[str | dict[str, dict[str, str]]] = "config/tasks.yaml"

    agents: ClassVar[list[BaseAgent]] = []
    tasks: ClassVar[list[Task]] = []

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self._leftover_manager: Agent | None = None

    @agent
    def leftover_manager(self) -> Agent:
        if self._leftover_manager is None:
            config = self.agents_config
            assert isinstance(config, dict), (
                "agents_config not loaded — check config/agents.yaml exists"
            )
            agent_config = config["leftover_manager"]
            self._leftover_manager = Agent(
                role=agent_config["role"],
                goal=agent_config["goal"],
                backstory=agent_config["backstory"],
                llm=self._llm,
                max_iter=agent_config.get("max_iter", 3),
                allow_delegation=agent_config.get("allow_delegation", False),
                verbose=agent_config.get("verbose", False),
            )
        return self._leftover_manager

    @task
    def leftover_task(self) -> Task:
        config = self.tasks_config
        assert isinstance(config, dict), (
            "tasks_config not loaded — check config/tasks.yaml exists"
        )
        task_config = config["leftover_task"]
        return Task(
            description=task_config["description"],
            expected_output=task_config["expected_output"],
            agent=self.leftover_manager(),  # type: ignore[call-arg]
        )
