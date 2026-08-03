"""Notebook cell 15: "Built-in Agent Types" / code execution.

Classic (what the notebook ships):
    assistant  = AssistantAgent(name="assistant", llm_config={...})
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="coding")},
    )
    user_proxy.initiate_chat(assistant, message="Plot a sine wave ... save as sine_wave.png",
                             max_turns=4)

The assistant writes Python as prose in a message; the proxy scrapes the
fenced code block out of the transcript, writes it to disk and executes it
on the host; the result goes back as another message; up to four rounds of
that if the code fails.

Neither class exists in v1.0. Grepped the whole package: zero occurrences
of AssistantAgent or UserProxyAgent. There is one ``Agent`` class, and
Classic's UserProxyAgent split along the seam it was conflating -
human-in-the-loop is now ``hitl_hook``, code execution is a tool.

v1.0 (here): ONE agent holding ONE typed tool.

The two-agent loop was scaffolding for a limitation that no longer
applies. It existed because the model could only emit code as text, so
something had to read the transcript and run what it found. Modern models
emit structured tool calls, so the framework executes and feeds the result
back inside the same loop. The retry behaviour survives - a tool that
raises sends the error back to the model - it just no longer needs a
second participant. Measured: one self-correction cycle takes the turn
from 2 model calls to 3, and prompt tokens from ~1,760 to ~4,460, since
the whole conversation plus the tool schema re-rides on every call.

This script goes one step further than swapping the executor for a
sandbox. The sine wave was a stand-in for "the model needs to do something
computational"; because we know what that something IS, it can be a typed
function. The model then chooses ARGUMENTS, not code, and there is nothing
arbitrary left to sandbox. ``SandboxCodeTool`` with a Docker backend is
the right reach when the task genuinely is not enumerable - AG2 v1.0
refuses to run model-written code on a local backend by default, which is
exactly what the notebook does.

No free-text expression parameter. ``function`` is a Literal picklist and
the bounds are floats, so nothing model-supplied is ever parsed or
evaluated. ``filename`` is the one string the model controls.

The .png constraint lives only in ``_safe_output_path``, deliberately not
in ``plot_wave``'s Args block. Stating it there makes the model comply
silently at schema level; omitting it forces the error through the tool
boundary, where the model recovers and reports the substitution. The
second request asks for .jpg to exercise that path on every run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from ag2 import Agent, tool
from matplotlib.figure import Figure

from helpers.config import build_config
from helpers.metrics import print_usage

matplotlib.use("Agg")  # headless backend; no GUI thread in an async context

OUTPUT_DIR = Path(__file__).parent / "plots"

WaveFunction = Literal["sin", "cos", "tan"]

_WAVES = {"sin": np.sin, "cos": np.cos, "tan": np.tan}

REQUESTS = (
    "Plot a sine wave from -2*pi to 2*pi and save it as sine_wave.png. "
    "Tell me where you saved it.",
    "Plot a cosine wave from 0 to 4*pi and save it as cosine_wave.jpg. "
    "Tell me where you saved it.",
)


def _safe_output_path(filename: str) -> Path:
    """Resolve a model-supplied filename to a path inside OUTPUT_DIR.

    The model picks this string, so it is untrusted input at a filesystem
    boundary. Reject anything that is not a bare ``name.png`` stem rather
    than trying to sanitise separators out of it.
    """
    candidate = Path(filename)
    if candidate.name != filename or candidate.suffix != ".png":
        raise ValueError(f"filename must be a bare '*.png' name, got {filename!r}")
    return OUTPUT_DIR / candidate.name


@tool
def plot_wave(function: WaveFunction, start: float, stop: float, filename: str) -> str:
    """Plot a trigonometric wave over a range and save it as a PNG.

    Args:
        function: Which wave to plot - one of "sin", "cos", "tan".
        start: Left edge of the x range, in radians.
        stop: Right edge of the x range, in radians.
        filename: Output file name.
    """
    if stop <= start:
        raise ValueError(f"stop ({stop}) must be greater than start ({start})")

    path = _safe_output_path(filename)
    OUTPUT_DIR.mkdir(exist_ok=True)

    x = np.linspace(start, stop, 1000)
    y = _WAVES[function](x)

    figure = Figure(figsize=(8, 4))
    axes = figure.subplots()
    axes.plot(x, y)
    axes.set_title(f"{function}(x) from {start:.2f} to {stop:.2f}")
    axes.set_xlabel("x (radians)")
    axes.set_ylabel(f"{function}(x)")
    axes.grid(visible=True, alpha=0.3)
    figure.savefig(path, dpi=100, bbox_inches="tight")

    return f"Saved {function} wave over [{start:.4f}, {stop:.4f}] to {path}"


async def main() -> None:
    agent = Agent(
        "plotter",
        "You produce plots on request. Use the plot_wave tool; never describe code.",
        config=build_config(),
        tools=[plot_wave],
    )

    for request in REQUESTS:
        reply = await agent.ask(request)
        body = reply.body
        assert body is not None, "plotter produced no text body"
        print(f"\n[plotter]\n{body}")
        print_usage(await reply.usage(), label=request[:40])


if __name__ == "__main__":
    asyncio.run(main())
