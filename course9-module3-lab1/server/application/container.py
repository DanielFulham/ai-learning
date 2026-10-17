from fastmcp import FastMCP

from application.lab_server import build_server
from config import RESOURCES_DIR, ROOT_DIR, SERVER_NAME
from infra.local_file_store import LocalFileStore
from interfaces.file_store_interface import FileStoreInterface


def initialise_server(
    file_store: FileStoreInterface | None = None,
    resource_store: FileStoreInterface | None = None,
) -> FastMCP:
    """Composition root for the MCP server: wires the two file stores and builds the server.

    `file_store` is where the tools write and `resource_store` is where the resource
    template reads. Stateless: creates fresh instances every call. Both parameters
    default to the production concrete.
    """
    if file_store is None:
        file_store = LocalFileStore(ROOT_DIR)

    if resource_store is None:
        resource_store = LocalFileStore(RESOURCES_DIR)

    return build_server(file_store, resource_store, SERVER_NAME)
