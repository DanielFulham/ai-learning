from fastmcp import FastMCP

from interfaces.document_store_interface import DocumentStoreInterface


def build_server(document_store: DocumentStoreInterface) -> FastMCP:
    """Build the calculator MCP server, reading documents through `document_store`."""
    mcp = FastMCP(
        name="CalculatorMCPServer",
        instructions="""
            This server provides data analysis tools.
            Call get_average() to analyze numerical data.
        """,
    )

    @mcp.tool
    def add(a: int, b: int) -> int:
        """
        Add two integers together.

        Args:
            a (int): The first integer.
            b (int): The second integer.

        Returns:
            int: The sum of `a` and `b`.

        Example:
            >>> add(3, 5)
            8
        """
        return a + b

    @mcp.tool
    def subtract(a: int, b: int) -> int:
        """
        Subtract one integer from another.

        Args:
            a (int): The number to subtract from.
            b (int): The number to subtract.

        Returns:
            int: The result of `a - b`.

        Example:
            >>> subtract(10, 4)
            6
        """
        return a - b

    @mcp.resource("file:///endpoint/{name}")
    def return_template_document(name: str) -> str:
        """Read a document by name"""
        return f"Document contents of {name}"

    @mcp.resource("file://endpoint2/{name}")
    def read_document(name: str) -> str:
        """Read a document by name from the path directory"""
        text = document_store.read(name)
        if text is None:
            return f"Document '{name}' not found in path directory"
        return text

    @mcp.prompt(title="Code Review")
    def review_code(code: str) -> str:
        return f"Please review this code:\n\n{code}"

    return mcp
