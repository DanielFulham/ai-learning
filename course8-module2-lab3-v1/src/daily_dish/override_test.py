"""Override test — task-tools vs agent-tools when BOTH are set.

The load-bearing question for this lab: when an Agent has tools
declared AND a Task attached to that agent has its own tools declared,
does the task-level list OVERRIDE the agent-level list, or AUGMENT it?

The CrewAI docs assert override: "tasks with specific tools can override
an agent's default set for tailored task execution." This script closes
the loop by running it.

Setup:
- Agent has BOTH tools: [pdf_tool, serper_tool]
- Task 1 (retrieval) binds ONLY [pdf_tool]
- Task 2 (drafting) binds no tools

Prediction: if override holds, Task 1 will not invoke Serper even though
the agent "has" it. Query 4 (parking) is the critical test — Approach 1
escalated to Serper on this query, Approach 2 could not (no Serper
anywhere). If Approach 3 matches Approach 2's behaviour, override is
confirmed. If it matches Approach 1's behaviour, augmentation is
confirmed.
"""

from __future__ import annotations

from config import build_llm
from crewai import Agent, Crew, Process, Task
from crewai.crews.crew_output import CrewOutput
from fixtures import FIXTURE_QUERIES
from tools import build_pdf_search_tool, build_serper_tool


def build_crew() -> Crew:
    """Construct the override-test Daily Dish crew.

    The critical distinction from task_centric.py:
    - Agent gets BOTH tools attached
    - Task 1 binds only PDF
    - Task 2 binds nothing

    If task-level tools override agent-level tools during task execution,
    Task 1 will not have Serper available even though the agent "has" it.
    """
    llm = build_llm()
    pdf_tool = build_pdf_search_tool()
    serper_tool = build_serper_tool()

    customer_service_specialist = Agent(
        role="The Daily Dish Customer Service Specialist",
        goal=(
            "Follow a structured, multi-step process to answer customer "
            "questions about The Daily Dish. You have both FAQ search "
            "and web search available, but each task will specify which "
            "tool to use."
        ),
        backstory=(
            "You are an AI assistant for 'The Daily Dish' with access to "
            "both the restaurant's FAQ document and general web search. "
            "You operate under a strict task pipeline — each task will "
            "tell you which tool to use for that step. Follow the pipeline "
            "faithfully."
        ),
        tools=[pdf_tool, serper_tool],
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
    """Kick off the override-test crew against each fixture query.

    Watch stdout for tool selection on Query 4 in particular. If Serper
    fires, the override claim is wrong. If it doesn't, override is
    confirmed.
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
