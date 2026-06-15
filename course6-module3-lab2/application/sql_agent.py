from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from application.middleware import make_log_tool_call_middleware
from domain.models import AgentTrace

from typing import Sequence


_SYSTEM_PROMPT_TEMPLATE = """\
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.\
"""

_DEFAULT_TOP_K = 5


class SqlAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        dialect: str,
        trace: AgentTrace,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self._trace = trace
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(dialect=dialect, top_k=top_k)
        self._agent = create_agent(
            llm,
            tools,
            system_prompt=system_prompt,
            middleware=[make_log_tool_call_middleware(trace)],
        )

    def ask(self, question: str) -> str:
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        content = result["messages"][-1].content
        if not isinstance(content, str):
            raise TypeError(
                f"Expected str content from final AIMessage, got {type(content).__name__}"
            )
        return content

    @property
    def trace(self) -> AgentTrace:
        return self._trace