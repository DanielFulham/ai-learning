from pathlib import Path

SERVER_NAME = "File Operations MCP Server"

# The only folder the file tools can reach. It sits beside the server's own code
# so the tools cannot read or change that code (the lab used the current directory).
ROOT_DIR = Path(__file__).resolve().parent / "workspace"

# Where the server listens when it is run over HTTP (`python run_server.py http`).
HOST = "127.0.0.1"
PORT = 8000
