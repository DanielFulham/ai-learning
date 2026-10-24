from pathlib import Path

SERVER_NAME = "Permission-Aware MCP Server"

_SERVER_DIR = Path(__file__).resolve().parent

# The only folder the file tools can reach.
ROOT_DIR = _SERVER_DIR / "workspace"

# The policy file and the audit log. This folder sits beside ROOT_DIR, not inside it, so the
# file tools cannot read, overwrite or delete either file.
STATE_DIR = _SERVER_DIR / "state"
POLICY_FILE_NAME = "permissions.json"
AUDIT_LOG_FILE_NAME = "audit.log"
