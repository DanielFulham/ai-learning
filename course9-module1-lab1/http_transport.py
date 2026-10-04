import asyncio

import requests
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from helpers.common import run_lab_steps

HTTP_URL = "https://mcp.context7.com/mcp"
PREFACE_URL = "https://www.ibm.com/"


def http_preface() -> None:
    r = requests.get(PREFACE_URL, timeout=10)

    print(r.status_code)
    print(r.request.headers)
    print("request body:", r.request.body)


def build_http_client() -> Client:
    http_transport = StreamableHttpTransport(url=HTTP_URL)
    print(http_transport)
    return Client(http_transport)


def main() -> None:
    print("--- HTTP preface ---")
    http_preface()

    print("\n--- HTTP transport ---")
    asyncio.run(run_lab_steps(build_http_client()))


if __name__ == "__main__":
    main()