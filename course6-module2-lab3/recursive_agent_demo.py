from application.container import AgentStrategy, initialise


def main() -> None:
    agent = initialise(strategy=AgentStrategy.RECURSIVE)

    queries = [
        "Summarize this YouTube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english",
        "Show top 3 US trending videos with metadata and thumbnails",
        "Search YouTube for 'RAG explained' and get the metadata for the top result",
    ]

    for query in queries:
        print(f"\n--- Query: {query}")
        print(f"Response:\n{agent.run(query)}")


if __name__ == "__main__":
    main()