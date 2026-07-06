import asyncio
import os
import subprocess
import sys
import time

from parallel import ParallelState, app
from pathlib import Path
from shared import describe_graph, llm, render_graph_png


async def run_once(provider: str) -> tuple[float, float, str]:
    """Return (single_call_baseline, parallel_wall_clock, combined_output)."""
    describe_graph(app)
    png_path = Path(__file__).parent / f"graph_parallel_{provider}.png"
    render_graph_png(app, str(png_path))

    single_start = time.perf_counter()
    _ = await llm.ainvoke("Translate the following text to French:\n\nGood morning! I hope you have a wonderful day.")
    single_elapsed = time.perf_counter() - single_start

    input_state: ParallelState = {
        "text": "Good morning! I hope you have a wonderful day.",
        "translations": [],
        "combined_output": "",
    }

    start = time.perf_counter()
    result = await app.ainvoke(
        input_state,
        config={"configurable": {"thread_id": f"parallel-demo-{provider}"}},
    )
    elapsed = time.perf_counter() - start

    return single_elapsed, elapsed, result["combined_output"]


async def measure_provider() -> None:
    """Run for the provider currently in shared.llm and print timings."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    single, parallel_wall, combined = await run_once(provider)

    print(f"\n--- Provider: {provider} ---")
    print(f"Single call baseline (async): {single:.2f}s")
    print(f"Parallel wall-clock:          {parallel_wall:.2f}s")
    print("\nCOMBINED OUTPUT")
    print("===============")
    print(combined)


def compare_providers() -> None:
    """Spawn a subprocess per provider so each gets its own import-time llm."""
    for provider in ("anthropic", "openai"):
        print(f"\n=== Running with LLM_PROVIDER={provider} ===")
        env = os.environ.copy()
        env["LLM_PROVIDER"] = provider
        env["PARALLEL_MODE"] = "measure"
        subprocess.run([sys.executable, __file__], env=env, check=True)


if __name__ == "__main__":
    if os.environ.get("PARALLEL_MODE") == "measure":
        asyncio.run(measure_provider())
    else:
        compare_providers()