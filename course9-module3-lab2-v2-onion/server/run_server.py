import logging
import sys
import warnings

from application.container import initialise_server
from application.http_security import host_origin_options
from config import DEFAULT_TRANSPORT, HOST, PORT, ROOT_DIR

if __name__ == "__main__":
    # Suppress deprecation warnings and verbose FastMCP logging, as the lab does.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.getLogger("fastmcp").setLevel(logging.WARNING)

    transport = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRANSPORT
    server = initialise_server()

    if transport == "http":
        print(f"Starting HTTP MCP Server on http://{HOST}:{PORT}")
        print(f"Workspace roots: {ROOT_DIR}")
        server.run(transport="http", host=HOST, port=PORT, **host_origin_options())
    elif transport == "stdio":
        # No start-up lines here: on stdio, anything printed to stdout corrupts the MCP stream.
        server.run()
    else:
        print("Usage: python run_server.py [http|stdio]", file=sys.stderr)
        sys.exit(1)
