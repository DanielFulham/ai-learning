from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from application.container import initialise_server
from application.http_security import host_origin_options
from config import ALLOWED_HOSTS, ALLOWED_ORIGINS, HOST, PORT
from interfaces.file_store_interface import FileStoreInterface

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "TEST_CLIENT", "version": "0"},
    },
}
_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _post_initialize(extra_headers: dict[str, str]) -> int:
    server = initialise_server(MagicMock(spec=FileStoreInterface))
    app = server.http_app(**host_origin_options())
    with TestClient(app, base_url=f"http://{HOST}:{PORT}") as client:
        response = client.post("/mcp", json=_INITIALIZE, headers={**_HEADERS, **extra_headers})
    return response.status_code


def test_the_checks_are_turned_on_with_the_servers_own_addresses() -> None:
    options = host_origin_options()
    assert options["host_origin_protection"] is True
    assert options["allowed_hosts"] == ALLOWED_HOSTS
    assert options["allowed_origins"] == ALLOWED_ORIGINS


def test_the_allowed_addresses_are_the_servers_own_loopback_address_and_localhost() -> None:
    assert ALLOWED_HOSTS == [f"{HOST}:{PORT}", f"localhost:{PORT}"]
    assert ALLOWED_ORIGINS == [f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"]


def test_a_request_with_no_origin_is_accepted() -> None:
    # Our own client is not a browser and sends no Origin header.
    assert _post_initialize({}) == 200


@pytest.mark.parametrize("origin", [f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"])
def test_a_request_from_the_servers_own_origin_is_accepted(origin: str) -> None:
    assert _post_initialize({"Origin": origin}) == 200


def test_a_request_from_a_foreign_origin_is_refused() -> None:
    assert _post_initialize({"Origin": "http://evil.example"}) == 403


def test_a_request_for_a_foreign_host_is_refused() -> None:
    assert _post_initialize({"Host": "evil.example"}) == 421


def test_a_request_for_localhost_is_accepted() -> None:
    assert _post_initialize({"Host": f"localhost:{PORT}"}) == 200
