import asyncio
import sys

from dotenv import load_dotenv

from application.container import initialise
from config import DEFAULT_SERVER_SCRIPT


async def main() -> None:
    load_dotenv()  # ANTHROPIC_API_KEY from a .env file, if there is one
    server = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER_SCRIPT
    print(f"Connecting to server: {server}")
    await initialise(server).run()


if __name__ == "__main__":
    asyncio.run(main())
