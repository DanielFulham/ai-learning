"""Shared metrics helpers for BeeAI agent runs."""

from beeai_framework.agents.requirement.types import RequirementAgentRunState


def print_run_metrics(state: RequirementAgentRunState) -> None:
    """Print token usage, cache hit rate, and USD cost for a completed agent run."""
    print(f"Agent iterations: {state.iteration}")
    print(
        f"Tokens: prompt={state.usage.prompt_tokens} "
        f"(cached {state.usage.cached_prompt_tokens}), "
        f"completion={state.usage.completion_tokens}, "
        f"total={state.usage.total_tokens}"
    )

    prompt_tokens = state.usage.prompt_tokens
    if prompt_tokens > 0:
        cache_rate = state.usage.cached_prompt_tokens / prompt_tokens
        print(f"Cache hit rate: {cache_rate:.1%}")
    else:
        print("Cache hit rate: n/a (no prompt tokens recorded)")

    print(f"Cost: ${state.cost.total_cost_usd:.4f}")


def print_preview(text: str, label: str = "Analysis preview", limit: int = 200) -> None:
    """Print a truncated preview of agent output text."""
    preview = text[:limit] + "..." if len(text) > limit else text
    print(f"\n{label}:\n{preview}")
