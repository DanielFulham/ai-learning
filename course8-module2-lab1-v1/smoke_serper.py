"""Serper connectivity smoke test — equivalent to notebook cells 14-20.

Not part of the pipeline; run once when setting up to prove the API key
works, then leave alone.
"""

from crewai_tools import SerperDevTool
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()  # Reads SERPER_API_KEY from .env; no os.environ[...] = ... needed

    search_tool = SerperDevTool()
    print(type(search_tool))

    search_query = "Latest Breakthroughs in machine learning"
    search_results = search_tool.run(search_query=search_query)

    print(f"Search Results for '{search_query}':\n")  # Fixed notebook's f-string bug
    print("Result keys:", list(search_results.keys()))


if __name__ == "__main__":
    main()
