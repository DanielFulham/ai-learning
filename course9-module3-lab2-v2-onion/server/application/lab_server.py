from fastmcp import FastMCP

from interfaces.file_store_interface import FileStoreInterface, PathOutsideRootError

_DENIED = "Error: Access denied - path outside workspace roots"


def _register_tools(mcp: FastMCP, file_store: FileStoreInterface) -> None:
    @mcp.tool()
    def read_file(filepath: str) -> str:
        """Read a file from the workspace directory."""
        try:
            kind = file_store.kind_of(filepath)
        except PathOutsideRootError:
            return _DENIED

        if kind == "missing":
            return f"Error: File not found: {filepath}"

        try:
            content = file_store.read_text(filepath)
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @mcp.tool()
    def write_file(filepath: str, content: str) -> str:
        """Write content to a file in the workspace directory."""
        try:
            file_store.write_text(filepath, content)
            return f"Successfully wrote {len(content)} characters to {filepath}"
        except PathOutsideRootError:
            return _DENIED
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @mcp.tool()
    def list_files(directory: str = ".") -> str:
        """List files in a directory within the workspace."""
        try:
            kind = file_store.kind_of(directory)
        except PathOutsideRootError:
            return _DENIED

        if kind == "missing":
            return f"Error: Directory not found: {directory}"

        if kind != "directory":
            return f"Error: Not a directory: {directory}"

        try:
            files = []
            for entry in file_store.list_entries(directory):
                file_type = "DIR" if entry["is_dir"] else "FILE"
                files.append(f"{file_type}: {entry['path']} ({entry['size']} bytes)")

            return "\n".join(files) if files else "Directory is empty"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    @mcp.tool()
    def analyze_code(code: str, focus: str = "quality") -> str:
        """Analyze code focusing on specified aspect.

        In a full MCP implementation with bidirectional communication,
        this tool would send a sampling/createMessage JSON-RPC request
        to the client. For this educational lab, we return a message
        indicating where sampling would occur.
        """
        return f"""[SAMPLING TRIGGER]
This tool would send a sampling/createMessage request to the client:

{{
  'method': 'sampling/createMessage',
  'params': {{
    'messages': [{{'role': 'user', 'content': {{
      'type': 'text',
      'text': 'Analyze this code for {focus}:\\n{code[:50]}...'
    }}}}}}],
    'maxTokens': 500
  }}
}}

The client would:
1. Show approval dialog to user
2. If approved, call LLM with the prompt
3. Return LLM response to server
4. Server would use response to complete analysis

Note: Full bidirectional sampling requires low-level MCP SDK.
This simplified version demonstrates the concept."""


def _register_resources(mcp: FastMCP, file_store: FileStoreInterface) -> None:
    @mcp.resource("file://workspace/{filename}")
    def get_workspace_file(filename: str) -> str:
        """Read a file from the workspace as a resource."""
        try:
            kind = file_store.kind_of(filename)
        except PathOutsideRootError:
            raise ValueError("Access denied - path outside workspace roots") from None

        if kind == "missing":
            raise ValueError(f"File not found: {filename}")

        return file_store.read_text(filename)


def _register_prompts(mcp: FastMCP) -> None:
    @mcp.prompt()
    def review_code(filename: str) -> str:
        """Generate a prompt to review code from a file."""
        return f"""Please review the code in file '{filename}' and provide:

1. A summary of what the code does
2. Potential bugs or issues
3. Security concerns
4. Suggestions for improvements
5. Code quality assessment

Focus on readability, maintainability, and best practices."""

    @mcp.prompt()
    def analyze_security(filename: str) -> str:
        """Generate a prompt to analyze security of a file."""
        return f"""Perform a security analysis of '{filename}' focusing on:

1. Input validation and sanitization
2. Authentication and authorization checks
3. Potential injection vulnerabilities
4. Data exposure risks
5. Error handling security

Provide specific line numbers and remediation suggestions."""


def build_server(file_store: FileStoreInterface, name: str) -> FastMCP:
    """Build the lab's HTTP file server around a file store."""
    mcp = FastMCP(name)
    _register_tools(mcp, file_store)
    _register_resources(mcp, file_store)
    _register_prompts(mcp)
    return mcp
