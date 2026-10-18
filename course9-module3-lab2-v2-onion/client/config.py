import sys
from pathlib import Path

# The OpenAI chat model the host app talks to. The key comes from OPENAI_API_KEY in the
# environment (a `.env` file in this folder is loaded by the host entry point).
MODEL_ID = "gpt-4o-mini"

# The MCP server both apps connect to when none is given on the command line. As in the lab,
# this is the server's base address; the client adds the /mcp endpoint.
DEFAULT_SERVER = "http://127.0.0.1:8000"
MCP_ENDPOINT = "/mcp"

# Shown in the page header, as in the lab. It is display text only: the folder the server
# can reach is set in the server's own config.
DEFAULT_ROOTS_DISPLAY = "workspace"

# Where the two Gradio pages listen.
UI_HOST = "127.0.0.1"
GUI_PORT = 7861
HOST_PORT = 7862

# If a server script is given instead of a URL, it is launched over stdio. The server is its own
# app with its own virtual environment, so it is launched with that environment's interpreter
# (this app's environment does not have `fastmcp`), and told to use stdio.
_SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
_VENV_PYTHON = Path("Scripts", "python.exe") if sys.platform == "win32" else Path("bin", "python")
SERVER_COMMAND = str(_SERVER_DIR / "venv" / _VENV_PYTHON)
STDIO_SERVER_ARGS = ["stdio"]
