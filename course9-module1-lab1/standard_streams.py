"""Lab 41 V1 (canonical): the notebook's STDOUT / STDIN / STDERR background cells.

Standard library only. This is not MCP yet. It is the background for why the stdio
transport works: when `StdioTransport` starts a server as a subprocess, the server's
STDIN and STDOUT become the channel the protocol messages travel over, and its STDERR
stays separate (which is why a failed start-up is only visible there).

One deliberate divergence from the notebook: `sys.exit()` ends a script, whereas the
notebook's kernel survives it and keeps `name` for the next cell. Here the SystemExit
is caught so the same point can be shown.

Beyond the notebook: two happy paths at the end, because the notebook's cells are
mostly about errors. The second one (a parent and a child over pipes) is the shape of
the stdio transport.

Usage:
    1. python standard_streams.py
       (type a name; both streams show in the terminal)
    2. python standard_streams.py 2> stderr.txt
       (terminal shows STDOUT only; stderr.txt holds the error lines)
    3. "Dan" | python standard_streams.py > stdout.txt
       (name piped in, STDOUT goes to the file, terminal shows STDERR only)
"""

import json
import subprocess
import sys

# A toy child process that reads one JSON message per line from
# STDIN and writes one JSON reply per line to STDOUT. JSON-RPC shaped, not MCP.
CHILD_SOURCE = """
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    text = request["params"]["text"]
    reply = {"jsonrpc": "2.0", "id": request["id"], "result": {"echo": text}}
    sys.stdout.write(json.dumps(reply) + "\\n")
    sys.stdout.flush()
"""


def stdout_demo() -> None:
    my_code = "hello"
    print(my_code)

    written = sys.stdout.write("Hello")
    print()

    print(f"sys.stdout.write() returned: {written}")


def stdin_demo() -> str:
    name = input("What is your name? ")
    try:
        sys.exit("Notebook exited after input")
    except SystemExit as exit_signal:
        print(f"SystemExit caught: {exit_signal}", file=sys.stderr)
    print(name)
    return name


def stderr_demo() -> None:
    # normal output goes to stdout
    print("This is standard output (stdout)")

    # Error messages can be written to stderr
    print("This is an error message (stderr)", file=sys.stderr)

    # Example: catching a real error
    try:
        _ = 10 / 0
    except ZeroDivisionError as error:
        print(f"Error occurred: {error}", file=sys.stderr)


def happy_stdin_to_stdout(name: str) -> None:
    """Happy path 1: input arrives on STDIN, a result leaves on STDOUT, no errors."""
    print(f"Hello, {name}. Nothing went wrong, so STDERR stays quiet.")


def child_process_demo() -> None:
    """Happy path 2: a parent talks to a child process over its STDIN and STDOUT.

    The parent writes one line to the child's STDIN and reads one line back from its
    STDOUT. The stdio transport does the same with a server, except that it keeps the
    pipes open and exchanges many messages instead of one.
    """
    request = {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {"text": "hi"}}
    completed = subprocess.run(
        [sys.executable, "-c", CHILD_SOURCE],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    print(f"parent sent  : {json.dumps(request)}")
    print(f"child replied: {completed.stdout.strip()}")
    print(f"child stderr : {completed.stderr!r}")  # empty on the happy path


def main() -> None:
    print("--- STDOUT ---")
    stdout_demo()
    print("\n--- STDIN ---")
    name = stdin_demo()
    print("\n--- STDERR ---")
    stderr_demo()
    print("\n--- HAPPY PATH 1: STDIN to STDOUT ---")
    happy_stdin_to_stdout(name)
    print("\n--- HAPPY PATH 2: a parent and a child over pipes ---")
    child_process_demo()


if __name__ == "__main__":
    main()