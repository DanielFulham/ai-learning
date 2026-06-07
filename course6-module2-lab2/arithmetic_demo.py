"""Demo: arithmetic tool-calling agent."""

from dotenv import load_dotenv
load_dotenv()

from application.container import initialise
from application.tools.arithmetic import add, subtract, multiply


if __name__ == "__main__":
    agent = initialise(tools=[add, subtract, multiply])

    queries = [
        "one plus 2",
        "one - 2",
        "three times two",
    ]
    for q in queries:
        print(f"{q!r} → {agent.run(q)}")