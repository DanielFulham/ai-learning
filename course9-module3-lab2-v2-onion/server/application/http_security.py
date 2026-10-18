from typing import TypedDict

from config import ALLOWED_HOSTS, ALLOWED_ORIGINS


class HostOriginOptions(TypedDict):
    host_origin_protection: bool
    allowed_hosts: list[str]
    allowed_origins: list[str]


def host_origin_options() -> HostOriginOptions:
    """The Host and Origin checks to turn on when the server runs over HTTP."""
    return {
        "host_origin_protection": True,
        "allowed_hosts": ALLOWED_HOSTS,
        "allowed_origins": ALLOWED_ORIGINS,
    }
