from pathlib import Path

SERVER_NAME = "HTTP File Server"

# The only folder the file tools and the resource template can reach. It sits beside the
# server's own code, as in the lab.
ROOT_DIR = Path(__file__).resolve().parent / "workspace"

# Where the server listens over HTTP; FastMCP serves Streamable HTTP at /mcp.
HOST = "127.0.0.1"
PORT = 8000

# In HTTP mode only requests that name this server's own address are accepted: the Host header
# must be one of ALLOWED_HOSTS and, if the request has an Origin header (a browser sends one),
# it must be one of ALLOWED_ORIGINS. FastMCP leaves these checks off by default.
ALLOWED_HOSTS = [f"{HOST}:{PORT}", f"localhost:{PORT}"]
ALLOWED_ORIGINS = [f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"]

# The transport used when `run_server.py` is given no argument: "http" or "stdio".
DEFAULT_TRANSPORT = "http"
