"""Approach 2 — Task-Centric Tools (the Specialist Pipeline).

The Customer Service Specialist has no tools of its own. Instead, each
task binds the specific tool it needs:

- `faq_search_task` binds PDFSearchTool — retrieval only, no synthesis
- `response_drafting_task` binds no tools — synthesis only, using the
  previous task's context

The agent surrenders tool-selection reasoning entirely and follows a
deterministic pipeline instead.

Findings target: token cost per query vs Approach 1 on identical fixtures.
Task-centric prediction — cheaper per-query because each task's prompt
only carries its own tool description, but a second LLM call at the
drafting task boundary offsets some of the saving.
"""

from __future__ import annotations

from config import build_llm
from crewai import Agent, Crew, Process, Task
from crewai.crews.crew_output import CrewOutput
from fixtures import FIXTURE_QUERIES
from tools import build_pdf_search_tool


def build_crew() -> Crew:
    """Construct the task-centric Daily Dish crew.

    Single agent with NO tools attached. Two sequential tasks:
    - Task 1 (FAQ search) binds PDFSearchTool
    - Task 2 (response drafting) binds no tools, inherits context

    verbose=True on the Agent surfaces task-by-task execution in stdout —
    the load-bearing observability signal for the comparison against
    Approach 1.
    """
    llm = build_llm()
    pdf_tool = build_pdf_search_tool()

    customer_service_specialist = Agent(
        role="The Daily Dish Customer Service Specialist",
        goal=(
            "Follow a structured, multi-step process to answer customer "
            "questions about The Daily Dish. You will be given specific "
            "tasks in sequence — search first, then draft the response."
        ),
        backstory=(
            "You are an AI assistant for 'The Daily Dish' operating under "
            "a strict pipeline. Unlike a generalist assistant, you do not "
            "choose your own tools — each task specifies which tool (if any) "
            "to use. Your role is to execute the pipeline faithfully and "
            "produce a polished response at the end."
        ),
        tools=[],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    faq_search_task = Task(
        description=(
            "Search the Daily Dish FAQ document for information relevant "
            "to the following customer query: '{customer_query}'. "
            "Return only the relevant content from the FAQ — do not "
            "compose a customer-facing response yet."
        ),
        expected_output=(
            "Relevant excerpts from the FAQ document that address the "
            "customer's query. Preserve the source content faithfully."
        ),
        agent=customer_service_specialist,
        tools=[pdf_tool],
    )

    response_drafting_task = Task(
        description=(
            "Using the FAQ excerpts retrieved in the previous task, "
            "compose a clear, friendly, well-formatted response to the "
            "customer query: '{customer_query}'. Do not invent details "
            "beyond what the previous task retrieved."
        ),
        expected_output=(
            "A comprehensive and well-formatted answer to the customer's "
            "query, grounded in the FAQ excerpts from the previous task."
        ),
        agent=customer_service_specialist,
    )

    return Crew(
        agents=[customer_service_specialist],
        tasks=[faq_search_task, response_drafting_task],
        process=Process.sequential,
        verbose=False,
    )


def run_fixture_queries() -> None:
    """Kick off the task-centric crew against each fixture query and
    print per-query results with delta token usage. Same fixture list
    as agent_centric.py, both sourced from fixtures.py — apples-to-apples
    with Approach 1.
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
