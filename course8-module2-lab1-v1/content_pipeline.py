from crewai import LLM, Agent, Crew, Process, Task
from crewai.crews.crew_output import CrewOutput
from crewai.types.usage_metrics import UsageMetrics
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="anthropic/claude-haiku-4-5", max_tokens=2000)

research_agent = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge information and insights on any subject with comprehensive analysis",
    backstory=(
        "You are an expert researcher with extensive experience in gathering, "
        "analyzing, and synthesizing information across multiple domains. "
        "Your analytical skills allow you to quickly identify key trends, "
        "separate fact from opinion, and produce insightful reports on any topic. "
        "You excel at finding reliable sources and extracting valuable information efficiently."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[SerperDevTool()],
    max_iter=3,
)

writer_agent = Agent(
    role="Tech Content Strategist",
    goal="Craft well-structured and engaging content based on research findings",
    backstory=(
        "You are a skilled content strategist known for translating "
        "complex topics into clear and compelling narratives. Your writing makes "
        "information accessible and engaging for a wide audience."
    ),
    verbose=True,
    llm=llm,
    allow_delegation=False,
    max_iter=3,
)

social_agent = Agent(
    role="Social Media Strategist",
    goal="Generate engaging social media snippets based on the full article",
    backstory=(
        "A digital storyteller who excels at crafting compelling posts "
        "to drive engagement and traffic."
    ),
    verbose=True,
    llm=llm,
    allow_delegation=False,
    max_iter=3,
)

research_task = Task(
    description=(
        "Analyze the major {topic}, identifying key trends and technologies. "
        "Provide a detailed report on their potential impact."
    ),
    agent=research_agent,
    expected_output=(
        "A 400-word summary of {topic} covering 3-4 key trends, "
        "each with one emerging technology and its potential impact. "
        "Bullet points acceptable."
    ),
)

writer_task = Task(
    description=(
        "Create an engaging blog post based on the research findings about {topic}. "
        "Tailor the content for a tech-savvy audience, ensuring clarity and interest."
    ),
    agent=writer_agent,
    expected_output=(
        "A 4-paragraph blog post on {topic}, written clearly and engagingly for tech enthusiasts."
    ),
)

social_task = Task(
    description=(
        "Summarize the blog post about {topic} into 2-3 engaging social "
        "media posts suitable for platforms like LinkedIn or Twitter. "
        "Make sure the tone is informative, professional, and encourages "
        "further reading."
    ),
    agent=social_agent,
    expected_output=(
        "2-3 short social posts (each under 280 characters). "
        "No preamble, no commentary — just the posts, separated by blank lines."
    ),
)

crew = Crew(
    agents=[research_agent, writer_agent, social_agent],
    tasks=[research_task, writer_task, social_task],
    process=Process.sequential,
    verbose=True,
)


def main() -> None:
    result = crew.kickoff(inputs={"topic": "Latest Generative AI breakthroughs"})
    assert isinstance(result, CrewOutput), (
        f"Expected CrewOutput, got {type(result).__name__} — streaming enabled unexpectedly?"
    )

    print("Result type:", type(result).__name__)

    tasks_outputs = result.tasks_output

    print("\n--- Research task ---")
    print("Description:", tasks_outputs[0].description)
    print("Agent:", tasks_outputs[0].agent)
    print("Raw output:\n", tasks_outputs[0].raw)

    print("\n--- Writer task ---")
    print("Description:", tasks_outputs[1].description)
    print("Agent:", tasks_outputs[1].agent)
    print("Raw output:\n", tasks_outputs[1].raw)

    print("\n--- Social task ---")
    print("Description:", tasks_outputs[2].description)
    print("Agent:", tasks_outputs[2].agent)
    print("Raw output:\n", tasks_outputs[2].raw)

    usage = result.token_usage
    assert isinstance(usage, UsageMetrics)
    print("\n--- Token usage ---")
    print(f"Total:      {usage.total_tokens}")
    print(f"Prompt:     {usage.prompt_tokens} (of which cached: {usage.cached_prompt_tokens})")
    print(f"Completion: {usage.completion_tokens}")
    print(f"Requests:   {usage.successful_requests}")

    print(
        f"Messages: {len(tasks_outputs[0].messages)} (research), "
        f"{len(tasks_outputs[1].messages)} (writer), "
        f"{len(tasks_outputs[2].messages)} (social)"
    )


if __name__ == "__main__":
    main()