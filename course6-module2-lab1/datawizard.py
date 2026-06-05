from dotenv import load_dotenv

from application.container import initialise
from application.interfaces.data_wizard_agent_interface import DataWizardAgentInterface


# Module-level singleton — initialised once, reused across queries.
_agent: DataWizardAgentInterface | None = None


def main() -> None:
    """CLI entry point. Prompts the user for queries until they exit."""
    global _agent
    
    load_dotenv()
    
    if _agent is None:
        _agent = initialise()

    print("\n📊 Ask questions about your datasets (type 'exit' to quit):")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit", ""):
            print("Goodbye.")
            break
        
        response = _agent.ask(user_input)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    main()