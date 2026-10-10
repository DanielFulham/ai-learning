import sys

from application.container import initialise_server
from config import HOST, PORT

if __name__ == "__main__":
    # No startup banner of our own: on stdio, anything printed to stdout corrupts the MCP stream.
    server = initialise_server()
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        server.run(transport="http", host=HOST, port=PORT)
    else:
        server.run()
