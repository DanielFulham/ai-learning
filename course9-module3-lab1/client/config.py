import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
_VENV_PYTHON = Path("Scripts", "python.exe") if sys.platform == "win32" else Path("bin", "python")

# The server is its own app with its own virtual environment, so by default it is launched
# with that environment's interpreter (this app's environment does not have `fastmcp`).
SERVER_COMMAND = str(_SERVER_DIR / "venv" / _VENV_PYTHON)

# The server script launched over stdio when no server is given on the command line.
DEFAULT_SERVER_SCRIPT = str(_SERVER_DIR / "run_server.py")
