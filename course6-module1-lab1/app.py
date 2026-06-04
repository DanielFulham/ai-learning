"""Course 6 Module 1 — LangChain tool-calling lab entry point.

Each section is a standalone exercise in sections/sX_*.py. This file dispatches
to one section or runs them all sequentially.

Usage:
    python app.py                       # run all sections in order
    python app.py --section single      # run section 2 (single-tool agent)
    python app.py --list                # print available sections

Sections:
    direct      — Direct tool tests (no LLM)
    single      — Single-tool agent (GDP, 'two and 30', negative numbers)
    introspect  — Tool introspection and direct-invoke noise tests
    four-tool   — Four-tool math agent with deliberate-bug subtract
    harness     — Test harness with corrected subtract
    wikipedia   — Wikipedia tool + population × 0.75
    power       — Power tool (typed args, modernised final exercise)
"""
import argparse

from sections import (
    s1_direct_tool_tests,
    s2_single_tool_agent,
    s3_tool_introspection,
    s4_four_tool_agent,
    s5_harness,
    s6_wikipedia,
    s7_power_tool,
)

SECTIONS = {
    "direct":     s1_direct_tool_tests.run,
    "single":     s2_single_tool_agent.run,
    "introspect": s3_tool_introspection.run,
    "four-tool":  s4_four_tool_agent.run,
    "harness":    s5_harness.run,
    "wikipedia":  s6_wikipedia.run,
    "power":      s7_power_tool.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--section",
        choices=list(SECTIONS) + ["all"],
        default="all",
        help="Section to run (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sections and exit",
    )
    args = parser.parse_args()

    if args.list:
        for name in SECTIONS:
            print(name)
        return

    targets = list(SECTIONS.items()) if args.section == "all" else [(args.section, SECTIONS[args.section])]

    for name, run_fn in targets:
        print(f"\n{'#' * 70}")
        print(f"# Section: {name}")
        print(f"{'#' * 70}\n")
        run_fn()


if __name__ == "__main__":
    main()