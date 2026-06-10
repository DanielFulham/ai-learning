"""Domain objects representing the trace of an agent run.

An AgentTrace captures everything that happened during a single call to
DataVizAgent.run_with_trace — the original query, every tool call the LLM
made (with code and result), and the final natural-language answer.

These are plain dataclasses with no framework dependencies. They exist so
the application can return structured trace data instead of leaking
LangChain message objects across the seam. Consumers (demos, tests,
future observability layers) read these without knowing the agent is
built on LangGraph.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCallRecord:
    """A single tool invocation within an agent run."""

    name: str
    args: dict[str, object]
    result: str


@dataclass(frozen=True)
class AgentTrace:
    """The full record of one DataVizAgent.run_with_trace call."""

    query: str
    tool_calls: list[ToolCallRecord]
    final_answer: str