from collections.abc import Mapping
from types import MappingProxyType

from domain.risk_level import RiskLevel

TOOL_RISKS: Mapping[str, RiskLevel] = MappingProxyType(
    {
        "read_file": RiskLevel.LOW,
        "write_file": RiskLevel.MEDIUM,
        "delete_file": RiskLevel.HIGH,
        "execute_command": RiskLevel.CRITICAL,
    }
)
