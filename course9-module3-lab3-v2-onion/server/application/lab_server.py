import json
from collections.abc import Awaitable, Callable, Mapping

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from mcp_types import InputRequiredResult

from application.input_required import approval_request, outcome_from
from application.permission_gate import PermissionGate
from application.policy_book import PolicyBook
from domain.gate_decision import AskUser, RunNow
from domain.gate_result import Completed, GateResult
from domain.tool_risks import TOOL_RISKS
from interfaces.audit_log_interface import AuditLogInterface
from interfaces.file_store_interface import FileStoreInterface

Guard = Callable[
    [Context, str, Mapping[str, object], Callable[[], str]],
    Awaitable[str | InputRequiredResult],
]


def _describe(tool: str, summary: str) -> str:
    return f"{summary} (Risk: {TOOL_RISKS[tool].value.upper()})"


def _unwrap(result: GateResult) -> str:
    """Return the tool's text, or raise so the client receives an error result."""
    if isinstance(result, Completed):
        return result.text
    raise ToolError(result.reason)


def _make_guard(gate: PermissionGate) -> Guard:
    async def guard(
        ctx: Context,
        name: str,
        arguments: Mapping[str, object],
        action: Callable[[], str],
    ) -> str | InputRequiredResult:
        """Run a tool or resource read through the gate, asking by returning input-required.

        Round one carries no answer, so an ask policy returns the question with a token bound to
        these arguments. The client answers and retries; round two reads the answer and the
        token back and the gate decides again.
        """
        decision = gate.decide(
            name, arguments, outcome_from(ctx.input_responses), ctx.request_state
        )
        if isinstance(decision, RunNow):
            return _unwrap(gate.execute(name, arguments, action))
        if isinstance(decision, AskUser):
            return InputRequiredResult(
                input_requests=approval_request(decision.message), request_state=decision.token
            )
        raise ToolError(decision.reason)

    return guard


def _register_tools(mcp: FastMCP, file_store: FileStoreInterface, guard: Guard) -> None:
    @mcp.tool(
        description=_describe("read_file", "Read a file from the data directory."),
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def read_file(filepath: str, ctx: Context) -> str | InputRequiredResult:
        return await guard(
            ctx, "read_file", {"filepath": filepath}, lambda: file_store.read_text(filepath)
        )

    @mcp.tool(
        description=_describe("write_file", "Write content to a file in the data directory."),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def write_file(filepath: str, content: str, ctx: Context) -> str | InputRequiredResult:
        def action() -> str:
            file_store.write_text(filepath, content)
            return f"Successfully wrote to {filepath}"

        return await guard(ctx, "write_file", {"filepath": filepath, "content": content}, action)

    @mcp.tool(
        description=_describe("delete_file", "Delete a file from the data directory."),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def delete_file(filepath: str, ctx: Context) -> str | InputRequiredResult:
        def action() -> str:
            file_store.delete(filepath)
            return f"Successfully deleted {filepath}"

        return await guard(ctx, "delete_file", {"filepath": filepath}, action)

    @mcp.tool(
        description=_describe(
            "execute_command", "Execute a system command. For security, this is a simulation only."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=True,
        ),
    )
    async def execute_command(command: str, ctx: Context) -> str | InputRequiredResult:
        def action() -> str:
            return (
                f"Simulated execution of command: {command}\n"
                "(Actual execution disabled for security)"
            )

        return await guard(ctx, "execute_command", {"command": command}, action)


def _register_resources(
    mcp: FastMCP, audit_log: AuditLogInterface, policy_book: PolicyBook, guard: Guard
) -> None:
    @mcp.resource("file://audit/log")
    async def get_audit_log(ctx: Context) -> str | InputRequiredResult:
        """Get the audit log of all operations."""
        return await guard(
            ctx,
            "read_audit_log",
            {},
            lambda: audit_log.read_text() or "No audit log entries yet.",
        )

    def permissions_text() -> str:
        policies = {tool: policy.value for tool, policy in policy_book.all_policies().items()}
        return json.dumps(policies, indent=2)

    @mcp.resource("file://config/permissions")
    async def get_permissions_config(ctx: Context) -> str | InputRequiredResult:
        """Get the current permissions configuration."""
        return await guard(ctx, "read_permissions", {}, permissions_text)


def _register_prompt(mcp: FastMCP) -> None:
    @mcp.prompt()
    def security_review(operation: str, risk_level: str) -> str:
        """Generate a security review prompt for an operation."""
        return f"""Review this operation for security implications:

Operation: {operation}
Risk Level: {risk_level}

Please analyze:
1. What data or systems could be affected?
2. What are the potential security risks?
3. What safeguards should be in place?
4. Should this operation require user approval?
5. What should be logged for audit purposes?
"""


def build_server(
    file_store: FileStoreInterface,
    gate: PermissionGate,
    audit_log: AuditLogInterface,
    policy_book: PolicyBook,
    name: str,
) -> FastMCP:
    mcp = FastMCP(name)
    guard = _make_guard(gate)
    _register_tools(mcp, file_store, guard)
    _register_resources(mcp, audit_log, policy_book, guard)
    _register_prompt(mcp)
    return mcp
