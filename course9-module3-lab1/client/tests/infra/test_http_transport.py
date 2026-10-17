import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from infra.http_transport import HttpTransport


def _make_fake_http_client() -> MagicMock:
    @asynccontextmanager
    async def fake(url: str) -> AsyncIterator[tuple[str, str, object]]:
        yield "TEST_READ", "TEST_WRITE", lambda: "TEST_SESSION_ID"

    return MagicMock(side_effect=fake)


def test_open_connects_to_the_given_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_http_client()
    monkeypatch.setattr("infra.http_transport.streamable_http_client", fake)

    async def run() -> None:
        async with HttpTransport("http://TEST_HOST:1/mcp").open():
            pass

    asyncio.run(run())
    fake.assert_called_once_with("http://TEST_HOST:1/mcp")


def test_open_yields_only_the_read_and_write_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK yields a third value, the session-id getter. The session needs only two.
    monkeypatch.setattr("infra.http_transport.streamable_http_client", _make_fake_http_client())
    seen: list[object] = []

    async def run() -> None:
        async with HttpTransport("http://TEST_HOST:1/mcp").open() as streams:
            seen.extend(streams)

    asyncio.run(run())
    assert seen == ["TEST_READ", "TEST_WRITE"]
