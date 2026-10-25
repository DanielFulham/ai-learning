from enum import StrEnum


class Policy(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
