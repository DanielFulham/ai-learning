from fastmcp import FastMCP

from application.file_server import build_server
from config import ROOT_DIR, SERVER_NAME
from infra.local_file_store import LocalFileStore
from interfaces.file_store_interface import FileStoreInterface


def initialise_server(file_store: FileStoreInterface | None = None) -> FastMCP:
    """Composition root for the MCP server: wires a file store and builds the server.

    Stateless: creates fresh instances every call. The optional parameter
    defaults to the production concrete.
    """
    if file_store is None:
        file_store = LocalFileStore(ROOT_DIR)

    return build_server(file_store, SERVER_NAME)
