"""Per-agent token accounting that works inside a network channel.

Middleware reaches agents running under the network plugin: the handler in
`ag2/network/client/handlers.py` drives them through the ordinary public
`agent.ask(...)`, so the normal middleware chain runs. The `AgentReply` is
consumed into an envelope and never surfaces to a caller, but `on_llm_call`
sits below that.

`Agent._execute` builds a fresh middleware instance on every `ask()`, so a
counter held on the `BaseMiddleware` subclass resets between turns and
silently reports the last turn only. The long-lived `MiddlewareFactory` owns
the accumulator instead — the shape AG2's own `MetricsMiddleware`
(`ag2/middleware/builtin/metrics.py`, needs `ag2[metrics]`) uses.
"""

# Ruff UP037 strips quotes from forward refs under target-version = "py314"
# (PEP 649 defers annotation evaluation). This import keeps the module
# importable on earlier interpreters too — cheap insurance, no behaviour change.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ag2.annotations import Context
from ag2.events import BaseEvent, ModelResponse, Usage
from ag2.middleware import BaseMiddleware
from ag2.middleware.base import LLMCall, MiddlewareFactory

# Usage fields are `float | None`; absent counters read as None, not 0.
_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "thinking_tokens",
)


@dataclass(frozen=True, slots=True)
class AgentTokens:
    """Accumulated token counts for one agent across N model calls."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    thinking_tokens: int = 0

    def plus_tokens(self, other: AgentTokens) -> AgentTokens:
        """Combine two accumulated rows (per-call detail into a total)."""
        return replace(
            self,
            calls=self.calls + other.calls,
            **{name: getattr(self, name) + getattr(other, name) for name in _FIELDS},
        )

    def plus(self, usage: Usage) -> AgentTokens:
        values = {name: int(getattr(usage, name) or 0) for name in _FIELDS}
        return replace(
            self,
            calls=self.calls + 1,
            **{name: getattr(self, name) + values[name] for name in _FIELDS},
        )


class TokenLedger:
    """Shared accumulator. One per probe run, not one per agent."""

    def __init__(self) -> None:
        self._rows: dict[str, AgentTokens] = {}
        # Per-call detail. Totals alone are uninterpretable when two runs
        # make different numbers of calls — which is exactly what the
        # first real L40 run did (run B looped three times, run C once).
        self._calls: dict[str, list[AgentTokens]] = {}
        # What each agent actually saw, per call. Token counts tell you an
        # agent's context was small; only the events tell you what was in it.
        self._inputs: dict[str, list[str]] = {}

    def record(self, agent_name: str, usage: Usage) -> None:
        current = self._rows.get(agent_name, AgentTokens())
        self._rows[agent_name] = current.plus(usage)
        self._calls.setdefault(agent_name, []).append(AgentTokens().plus(usage))

    def per_call(self, agent_name: str) -> list[AgentTokens]:
        return list(self._calls.get(agent_name, []))

    def record_input(self, agent_name: str, digest: str) -> None:
        self._inputs.setdefault(agent_name, []).append(digest)

    def inputs(self, agent_name: str) -> list[str]:
        return list(self._inputs.get(agent_name, []))

    def counter_for(self, agent_name: str) -> TokenCounter:
        """Return a middleware factory that books this agent's calls here."""
        return TokenCounter(self, agent_name)

    def rows(self) -> dict[str, AgentTokens]:
        return dict(self._rows)

    def total(self) -> AgentTokens:
        out = AgentTokens()
        for row in self._rows.values():
            out = replace(
                out,
                calls=out.calls + row.calls,
                **{name: getattr(out, name) + getattr(row, name) for name in _FIELDS},
            )
        return out


class TokenCounter(MiddlewareFactory):
    """Long-lived factory. Holds the agent label; state lives on the ledger."""

    def __init__(self, ledger: TokenLedger, agent_name: str) -> None:
        self._ledger = ledger
        self._agent_name = agent_name

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return _CountingMiddleware(
            event,
            context,
            ledger=self._ledger,
            agent_name=self._agent_name,
        )


class _CountingMiddleware(BaseMiddleware):
    """Stateless by construction — one instance per `ask()`."""

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        ledger: TokenLedger,
        agent_name: str,
    ) -> None:
        super().__init__(event, context)
        self._ledger = ledger
        self._agent_name = agent_name

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        self._ledger.record_input(self._agent_name, _digest(events))
        response = await call_next(events, context)
        self._ledger.record(self._agent_name, response.usage)
        return response


def _digest(events: Sequence[BaseEvent], *, preview: int = 90) -> str:
    """One line describing what the model was handed on this call.

    `on_llm_call` receives the exact event sequence sent to the provider —
    the only direct evidence of what an agent could see.
    """
    parts: list[str] = []
    for event in events:
        text = str(getattr(event, "content", "") or "")
        if not text:
            text = " ".join(str(p) for p in getattr(event, "parts", []) or [])
        text = " ".join(text.split())
        label = type(event).__name__
        parts.append(f"{label}({text[:preview]}{'…' if len(text) > preview else ''})")
    return f"{len(events)} events: " + " | ".join(parts)
