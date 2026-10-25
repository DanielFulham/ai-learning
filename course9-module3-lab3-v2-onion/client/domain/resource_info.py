from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceInfo:
    """A resource the server offers at a fixed address, such as `file://audit/log`."""

    uri: str
    name: str
    description: str
