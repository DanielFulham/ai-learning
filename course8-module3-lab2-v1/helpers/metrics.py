"""Run metrics printer.

AG2 v1.0 reports usage via ``UsageReport``, aggregated from stream events.
There is no cost field anywhere on the surface - tokens only.

``Usage`` fields are ``float | None``, not ``int``, and any of them may be
None when a provider omits them, hence the explicit None handling rather
than assuming zeroes.
"""

from __future__ import annotations

from ag2.usage import UsageReport


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def _cache_rate(report: UsageReport) -> str:
    """Cache-read tokens as a share of total input tokens.

    Input = prompt + cache-read + cache-creation. Returns "-" only when the
    provider reports NO cache fields (both None) — a provider that
    explicitly reports zero gets a real 0%, since "measured and absent" and
    "not reported" are different facts.
    """
    total = report.total
    prompt = total.prompt_tokens or 0.0

    read_raw = total.cache_read_input_tokens
    created_raw = total.cache_creation_input_tokens
    if read_raw is None and created_raw is None:
        return "-"

    read = read_raw or 0.0
    created = created_raw or 0.0
    denominator = prompt + read + created
    if denominator == 0.0:
        return "-"
    return f"{read / denominator:.1%}"


def print_usage(report: UsageReport, *, label: str = "run") -> None:
    """Print a one-block usage summary for a completed run."""
    total = report.total
    print(f"\n--- usage: {label} ---")
    print(f"  prompt tokens        : {_fmt(total.prompt_tokens)}")
    print(f"  completion tokens    : {_fmt(total.completion_tokens)}")
    print(f"  total tokens         : {_fmt(total.total_tokens)}")
    print(f"  cache read           : {_fmt(total.cache_read_input_tokens)}")
    print(f"  cache creation       : {_fmt(total.cache_creation_input_tokens)}")
    print(f"  thinking tokens      : {_fmt(total.thinking_tokens)}")
    print(f"  cache read rate      : {_cache_rate(report)}")
    print(f"  llm calls (records)  : {len(report.records)}")

    if report.by_kind:
        print("  by kind:")
        for kind, usage in report.by_kind.items():
            prompt = _fmt(usage.prompt_tokens)
            completion = _fmt(usage.completion_tokens)
            print(f"    {kind:<20} prompt={prompt} completion={completion}")
