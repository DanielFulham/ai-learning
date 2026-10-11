import sys
from pathlib import Path

# Claude model identifier for API calls.
MODEL_ID = "claude-haiku-4-5"
MAX_TOKENS = 8192

# The server is launched as a subprocess over stdio, with the interpreter that is
# running this code (the venv's, not whatever is on PATH).
SERVER_COMMAND = sys.executable
DEFAULT_SERVER_SCRIPT = str(Path(__file__).resolve().parent.parent / "server" / "run_server.py")
