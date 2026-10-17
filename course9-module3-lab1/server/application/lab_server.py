from fastmcp import FastMCP

from interfaces.file_store_interface import FileStoreInterface


def _register_tools(mcp: FastMCP, file_store: FileStoreInterface) -> None:
    @mcp.tool()
    def echo(text: str) -> str:
        """Echo back the input text."""
        return f"Echo: {text}"

    @mcp.tool()
    def write_file(path: str, content: str) -> str:
        """Write content to a file."""
        file_store.write_text(path, content)
        return f"Successfully wrote to {path}"


def _register_resources(mcp: FastMCP, resource_store: FileStoreInterface) -> None:
    @mcp.resource("file://resources/{filename}")
    def read_resource_file(filename: str) -> str:
        """Read a file from the resources directory."""
        if not resource_store.exists(filename):
            return f"File not found: {filename}"
        return resource_store.read_text(filename)


def _register_prompts(mcp: FastMCP) -> None:
    @mcp.prompt()
    def review_file(filename: str) -> str:
        """Generate a prompt to review a file's contents."""
        return f"""Please review the file '{filename}' and provide:

1. A summary of its contents
2. Key points or sections
3. Any suggestions for improvement
4. Overall quality assessment

Use the appropriate tools to read the file if needed."""


def build_server(
    file_store: FileStoreInterface, resource_store: FileStoreInterface, name: str
) -> FastMCP:
    """Build the lab server: `file_store` is where the tools write, `resource_store` is
    where the resource template reads."""
    mcp = FastMCP(name)
    _register_tools(mcp, file_store)
    _register_resources(mcp, resource_store)
    _register_prompts(mcp)
    return mcp
