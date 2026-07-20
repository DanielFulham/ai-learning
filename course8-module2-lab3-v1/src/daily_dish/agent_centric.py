"""Approach 1 — Agent-Centric Tools (the Generalist).

The Daily Dish Inquiry Specialist gets both tools upfront; the LLM
chooses which to invoke per query based on its reasoning. This is the
baseline generalist approach, contrasted against the task-centric
approach in task_centric.py.

Findings target: token cost + tool-selection observability compared to
Approach 2 on the same fixture questions.
"""

from __future__ import annotations

from config import build_llm
from crewai import Agent, Crew, Process, Task
from crewai.crews.crew_output import CrewOutput
from fixtures import FIXTURE_QUERIES
from tools import build_pdf_search_tool, build_serper_tool


def build_crew() -> Crew:
    """Construct the agent-centric Daily Dish crew.

    Single agent with both PDF search and web search tools attached.
    Single broad task that lets the agent decide which tool to invoke
    based on the query. verbose=True on the Agent so tool-selection
    reasoning is observable in stdout — load-bearing for the findings
    comparison against the task-centric approach.
    """
    llm = build_llm()
    pdf_tool = build_pdf_search_tool()
    serper_tool = build_serper_tool()

    inquiry_specialist = Agent(
        role="The Daily Dish Inquiry Specialist",
        goal=(
            "Accurately answer customer questions about The Daily Dish "
            "restaurant. You must decide whether to use the restaurant's "
            "FAQ PDF or a web search to find the best answer."
        ),
        backstory=(
            "You are an AI assistant for 'The Daily Dish'. "
            "You have access to two tools: one for searching the "
            "restaurant's FAQ document and another for searching the web. "
            "Your job is to analyze the user's question and choose the "
            "most appropriate tool to find the information needed to "
            "provide a helpful response."
        ),
        tools=[pdf_tool, serper_tool],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    inquiry_task = Task(
        description=(
            "Answer the following customer query: '{customer_query}'. "
            "Analyze the question and use the tools at your disposal "
            "(PDF search or web search) to find the most relevant "
            "information. Synthesize the findings into a clear and "
            "friendly response."
        ),
        expected_output="A comprehensive and well-formatted answer to the customer's query.",
        agent=inquiry_specialist,
    )

    return Crew(
        agents=[inquiry_specialist],
        tasks=[inquiry_task],
        process=Process.sequential,
        verbose=False,
    )


def run_fixture_queries() -> None:
    """Kick off the agent-centric crew against each fixture query and
    print per-query results with delta token usage.
    """
    crew = build_crew()
    prev_prompt = 0
    prev_completion = 0
    for query in FIXTURE_QUERIES:
        print(f"\n{'=' * 72}")
        print(f"QUERY: {query}")
        print(f"{'=' * 72}")
        result = crew.kickoff(inputs={"customer_query": query})
        if not isinstance(result, CrewOutput):
            raise TypeError(f"Expected CrewOutput from kickoff, got {type(result).__name__}")
        print(f"\n--- ANSWER ---\n{result.raw}\n")
        if result.token_usage is not None:
            delta_prompt = result.token_usage.prompt_tokens - prev_prompt
            delta_completion = result.token_usage.completion_tokens - prev_completion
            print(
                f"--- USAGE (this query) ---\n"
                f"prompt_tokens={delta_prompt} "
                f"completion_tokens={delta_completion} "
                f"cached_prompt_tokens={result.token_usage.cached_prompt_tokens}\n"
            )
            prev_prompt = result.token_usage.prompt_tokens
            prev_completion = result.token_usage.completion_tokens


if __name__ == "__main__":
    run_fixture_queries()
