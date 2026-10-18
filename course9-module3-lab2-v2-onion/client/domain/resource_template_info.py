from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceTemplateInfo:
    """A resource URI pattern the server offers, such as `file://workspace/{filename}`."""

    name: str
    uri_template: str
    description: str
