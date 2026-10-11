import asyncio
from pathlib import Path

from fastmcp import Context, FastMCP
from fastmcp.server.elicitation import AcceptedElicitation
from pydantic import BaseModel

from application.mappers import to_file_entry, to_listing_item
from interfaces.file_store_interface import FileStoreInterface


class DocumentGeneratorSchema(BaseModel):
    """Pydantic model for documentation filename schema.

    Used in elicitation to capture user input for documentation file names.

    Attributes:
        file_path: Path of the file we want to generate document on
        name: The name of the documentation file to create
    """

    file_path: str
    name: str


_DOC_PROMPT_INTRO = (
    "You are an expert technical writer and documentation specialist. "
    "Create documentation for the following code file:"
)


def _register_tools(mcp: FastMCP, file_store: FileStoreInterface, chunk_delay: float) -> None:
    # One lock for every write and delete: a write spans several awaits, so two
    # overlapping calls on the same path could otherwise interleave their chunks.
    write_lock = asyncio.Lock()

    @mcp.tool()
    async def write_file(file_path: str, content: str, ctx: Context) -> str:
        """Create a new file with specified content.

        Creates parent directories if they don't exist. Writes content to the file
        using UTF-8 encoding.

        Args:
            file_path: Relative path where the file should be created
            content: Content to write to the file
            ctx: MCP context for logging

        Returns:
            Success message with file path

        Raises:
            Exception: If file creation fails (logged to context)
        """
        try:
            total = len(content)
            chunk_size = max(total // 10, 1)

            async with write_lock:
                # An empty first write creates the file even when the content is empty.
                file_store.write_text(file_path, "")
                for i in range(0, total, chunk_size):
                    file_store.write_text(file_path, content[i : i + chunk_size], append=True)
                    written = min(i + chunk_size, total)
                    await ctx.report_progress(
                        progress=written,
                        total=total,
                        message=f"Writing progress: {written}/{total}",
                    )
                    await asyncio.sleep(chunk_delay)

            await ctx.report_progress(progress=total, total=total, message="Write complete")
            await ctx.info(f"File written successfully to: {file_path}")
            return f"File written successfully to: {file_path}"
        except Exception as e:
            await ctx.error(f"Error creating file: {str(e)}")
            raise

    @mcp.tool()
    async def delete_file(file_path: str, ctx: Context) -> str:
        """Delete a file from the project directory.

        Validates that the path points to a file (not a directory) before deletion.

        Args:
            file_path: Relative path to the file to delete
            ctx: MCP context for logging

        Returns:
            Success or error message describing the operation result
        """
        try:
            kind = file_store.kind_of(file_path)
            if kind == "file":
                async with write_lock:
                    file_store.delete(file_path)
                await ctx.info(f"Successfully deleted file {file_path}")
                return f"Successfully deleted file {file_path}"
            if kind == "directory":
                await ctx.warning(f"Error: {file_path} is a directory, not a file")
                return f"Error: {file_path} is a directory, not a file"
            await ctx.warning(f"File not found: {file_path}")
            return f"File not found: {file_path}"
        except Exception as e:
            await ctx.error(f"Error deleting file: {str(e)}")
            return f"Error deleting file: {str(e)}"


def _register_resources(mcp: FastMCP, file_store: FileStoreInterface) -> None:
    @mcp.resource("file:///{file_name}")
    async def read_file_resource(file_name: str) -> dict[str, str]:
        """Read the content of a file as an MCP resource.

        Provides file content access through the MCP resource protocol using
        the file:/// URI scheme.

        Args:
            file_name: Relative path to the file to read

        Returns:
            Dictionary containing either file_content or error message
        """
        try:
            if file_store.kind_of(file_name) != "file":
                return {"error": f"Error: {file_name} is not a valid file"}
            return {"file_content": file_store.read_text(file_name)}
        except Exception as e:
            return {"error": f"Error reading file: {str(e)}"}

    @mcp.resource("dir://.")
    async def list_files_resource() -> dict[str, object]:
        """List files and directories in the server's root folder as an MCP resource.

        Provides directory listing through the MCP resource protocol using
        the dir:// URI scheme. Returns metadata for each item including name,
        path, type, size, and timestamps.

        Returns:
            Dictionary containing list of items with metadata, or error message
        """
        try:
            if file_store.kind_of(".") != "directory":
                return {"error": "The server root is not a valid directory"}
            items = [to_listing_item(to_file_entry(raw)) for raw in file_store.list_entries(".")]
            return {"items": items}
        except Exception as e:
            return {"error": f"Error listing files: {e}"}


def _register_prompts(mcp: FastMCP, file_store: FileStoreInterface) -> None:
    @mcp.prompt()
    async def code_review(file_path: str, ctx: Context) -> str:
        """Generate a prompt for code review and quality evaluation.

        Reads a code file and generates a prompt for Claude to perform a
        comprehensive code review.

        Args:
            file_path: Relative path to the code file to review
            ctx: MCP context for logging and communication

        Returns:
            Formatted prompt string for code review

        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        try:
            if file_store.kind_of(file_path) != "file":
                error_msg = f"Error: {file_path} is not a valid file"
                await ctx.warning(error_msg)
                raise FileNotFoundError(error_msg)

            current_code = file_store.read_text(file_path).strip()
            language = Path(file_path).suffix.lower()

            prompt = f"""You are an expert code editor. Review the following code quality.

File: {file_path}
Language (file suffix): {language or "unknown"}

Current code:
'''
{current_code}
'''

Provide a comprehensive evaluation of the code:

""".strip()
            await ctx.info("Successfully returned prompt")
            return prompt

        except Exception as e:
            await ctx.error(f"Error preparing code review prompt: {e}")
            raise

    @mcp.prompt()
    async def documentation_generator(ctx: Context) -> str:
        """Generate a prompt for creating code documentation.

        Reads a code file, elicits a documentation filename from the user,
        and generates a prompt for Claude to create comprehensive documentation.

        Args:
            ctx: MCP context for logging and elicitation

        Returns:
            Formatted prompt string for documentation generation

        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        try:
            result = await ctx.elicit(
                message="Please provide the subject file name and the documentation file name",
                response_type=DocumentGeneratorSchema,
            )

            if not isinstance(result, AcceptedElicitation):
                raise ValueError(f"Documentation request was not accepted: {result.action}")

            file_path = result.data.file_path
            if file_store.kind_of(file_path) != "file":
                error_msg = f"Error: {file_path} is not a valid file"
                await ctx.warning(error_msg)
                raise FileNotFoundError(error_msg)

            code = file_store.read_text(file_path).strip()
            language = Path(file_path).suffix.lower()
            doc_name = result.data.name

            prompt = f"""{_DOC_PROMPT_INTRO}

File: {file_path}
Language (file suffix): {language or "unknown"}

Current code:
'''
{code}
'''

Use MCP tools available to you to create the separate documentation file:
- **CRITICAL DETAIL: Name that separate document EXACTLY: {doc_name}**
- Add the .md suffix yourself if the name doesn't include it already""".strip()

            await ctx.info("Successfully returned prompt")
            return prompt

        except Exception as e:
            await ctx.error(f"Error generating code documentation prompt: {e}")
            raise


def build_server(file_store: FileStoreInterface, name: str, chunk_delay: float = 0.05) -> FastMCP:
    """Build the file operations MCP server around a file store."""
    mcp = FastMCP(name)
    _register_tools(mcp, file_store, chunk_delay)
    _register_resources(mcp, file_store)
    _register_prompts(mcp, file_store)
    return mcp
