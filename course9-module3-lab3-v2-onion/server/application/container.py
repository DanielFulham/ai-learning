from pathlib import Path

from fastmcp import FastMCP

import config
from application.lab_server import build_server
from application.permission_gate import PermissionGate
from application.policy_book import PolicyBook
from infra.file_audit_log import FileAuditLog
from infra.json_policy_store import JsonPolicyStore
from infra.local_file_store import LocalFileStore
from interfaces.audit_log_interface import AuditLogInterface
from interfaces.file_store_interface import FileStoreInterface
from interfaces.policy_store_interface import PolicyStoreInterface


def initialise_server(
    root_dir: Path = config.ROOT_DIR,
    state_dir: Path = config.STATE_DIR,
    file_store: FileStoreInterface | None = None,
    policy_store: PolicyStoreInterface | None = None,
    audit_log: AuditLogInterface | None = None,
    name: str = config.SERVER_NAME,
) -> FastMCP:
    """Wire the server. Stateless: every call builds a new server from new components.

    The file tools get `root_dir`. The policy file and the audit log live in `state_dir`, a
    different folder, so the tools cannot read, overwrite or delete either of them.
    """
    if file_store is None:
        resolved_root = root_dir.resolve()
        resolved_state = state_dir.resolve()
        if resolved_state == resolved_root or resolved_state.is_relative_to(resolved_root):
            raise ValueError("The state folder must be outside the root folder the tools can reach")
        file_store = LocalFileStore(root_dir)
    if policy_store is None:
        policy_store = JsonPolicyStore(state_dir / config.POLICY_FILE_NAME)
    if audit_log is None:
        audit_log = FileAuditLog(state_dir / config.AUDIT_LOG_FILE_NAME)

    policy_book = PolicyBook(policy_store)
    gate = PermissionGate(policy_book, audit_log)
    return build_server(file_store, gate, audit_log, policy_book, name)
