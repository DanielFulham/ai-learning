from pathlib import Path

SERVER_NAME = "lab-server"

# The folder the tools write into (the lab used the current directory at import).
ROOT_DIR = Path(__file__).resolve().parent / "workspace"

# The folder the resource template reads from: committed sample files, kept apart from
# the workspace the tools write into (the lab kept both under the current directory).
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"

# Where the server listens when it is run over HTTP (`python run_server.py http`).
HOST = "127.0.0.1"
PORT = 8000
