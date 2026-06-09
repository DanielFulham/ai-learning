from application.container import AgentStrategy, initialise


def main() -> None:
    agent = initialise(strategy=AgentStrategy.TWO_STEP_CHAIN)

    queries = [
        "Summarize this YouTube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english",
    ]

    for query in queries:
        print(f"\n--- Query: {query}")
        print(f"Response:\n{agent.run(query)}")


if __name__ == "__main__":
    main()