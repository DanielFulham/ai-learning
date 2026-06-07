"""Demo: tip-calculating tool-calling agent."""

from dotenv import load_dotenv
load_dotenv()

from application.container import initialise
from application.tools.tip import calculate_tip


if __name__ == "__main__":
    agent = initialise(tools=[calculate_tip])

    queries = [
        "How much should I tip on $60 at 20%?",
        "What's a 15% tip on a $87.43 bill?",
        "Hello, how are you?",
    ]
    for q in queries:
        print(f"{q!r} → {agent.run(q)}")