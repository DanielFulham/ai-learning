from dotenv import load_dotenv

from application.container import initialise


_DATABASE_URI = "sqlite:///file:Chinook.db?mode=ro&uri=true"


def main() -> None:
    load_dotenv()
    agent = initialise(database_uri=_DATABASE_URI)

    print("\nSQL agent ready. Type a question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print(f"\n{'=' * 80}")
        answer = agent.ask(question)
        print(answer)
        print(f"{'=' * 80}\n")

    print(f"\nSession captured {len(agent.trace)} tool calls:")
    for record in agent.trace.tool_calls:
        print(f"  {record.tool_name}({record.args}) -> {record.duration_ms:.1f}ms")


if __name__ == "__main__":
    main()