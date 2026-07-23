"""Human-in-the-loop approval handlers for AskPermissionRequirement."""

from typing import Any

from beeai_framework.tools import Tool


async def stdin_approval_handler(tool: Tool, tool_input: dict[str, Any]) -> bool:
    """Approval handler that shows tool arguments before asking on stdin.

    Framework doc example shadows input builtin with the tool-args param,
    which would break stdin. Renamed to tool_input to preserve the builtin.
    """
    print(f"\n[HITL] Agent requests: {tool.name}")
    print(f"[HITL] Arguments: {tool_input}")
    response = input("[HITL] Approve? [yes/no]: ").strip().lower()
    return response in {"yes", "y"}
