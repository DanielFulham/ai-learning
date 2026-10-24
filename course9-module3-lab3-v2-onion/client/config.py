import sys
from pathlib import Path

# The OpenAI chat model the host app talks to. The key comes from OPENAI_API_KEY in the
# environment (a `.env` file in this folder is loaded by the host entry point).
MODEL_ID = "gpt-4o-mini"

# Where the two Gradio pages listen.
UI_HOST = "127.0.0.1"
GUI_PORT = 7863
HOST_PORT = 7864

# The server is its own app with its own virtual environment, so it is launched with that
# environment's interpreter (this app's environment does not have `fastmcp`), over stdio.
_SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
_VENV_PYTHON = Path("Scripts", "python.exe") if sys.platform == "win32" else Path("bin", "python")
DEFAULT_SERVER_SCRIPT = _SERVER_DIR / "run_server.py"
DEFAULT_SERVER_PYTHON = _SERVER_DIR / "venv" / _VENV_PYTHON

# The two resources the Permissions tab shows. These addresses are part of the contract with
# the server, which defines them.
POLICY_RESOURCE_URI = "file://config/permissions"
AUDIT_LOG_RESOURCE_URI = "file://audit/log"

# How long a question the server asked waits for a person before it counts as a cancel, and how
# often a page looks for a waiting question.
APPROVAL_TIMEOUT_SECONDS = 300.0
POLL_SECONDS = 0.5
