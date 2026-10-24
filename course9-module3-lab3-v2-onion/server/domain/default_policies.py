from collections.abc import Mapping
from types import MappingProxyType

from domain.policy import Policy

DEFAULT_POLICIES: Mapping[str, Policy] = MappingProxyType(
    {
        "read_file": Policy.ALLOW,
        "write_file": Policy.ASK,
        "delete_file": Policy.DENY,
        "execute_command": Policy.DENY,
        "read_audit_log": Policy.ALLOW,
        "read_permissions": Policy.ALLOW,
    }
)
