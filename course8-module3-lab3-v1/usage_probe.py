"""L40 session 1 — token cost of channel registration and of adding a tool.

Three runs, one seed, one role, one model config:

    A. bare                 Agent(middleware=[counter]); await agent.ask(SEED)
    B. registered           same agent inside a two-step workflow channel
    C. registered + tool    B plus the real routing tool

B and C use `TransitionGraph.sequence` over [human, agent] — N-1 model calls
for N steps, so all three runs are one call each and directly comparable.

PREDICTIONS, recorded before the first run:
  1. B - A lands in 3,000-5,000 prompt tokens (five plugin tools plus
     per-turn adapter tools, extrapolated from F-L39-10).
  2. C - B lands well under F-L39-10's 1,273, because that figure was
     measured on a bare agent carrying no tool schemas at all.

RESULT (five runs, prompt tokens identical every time): A=39, B=2,454,
C=2,542. Prediction 1 wrong — 2,415, and it measures the plugin tools rather
than registration itself: with `attach_plugin=False` a channel agent costs
~76. Prediction 2 right at +88.

Usage:
    python usage_probe.py              # real provider
    python usage_probe.py --offline    # scripted client, wiring only, no spend
"""

import argparse
import asyncio
import sys

from ag2 import Agent
from ag2.config import ModelConfig, ModelProvider
from ag2.events import ModelMessage, ModelResponse, Usage
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    WORKFLOW_TYPE,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    TransitionGraph,
)
from ag2.testing import TestConfig

from helpers.config import MODEL, anthropic_config
from helpers.routing import record_consultation_status
from helpers.usage import AgentTokens, TokenLedger

AGENT_NAME = "responder"
HUMAN_NAME = "patient"

ROLE = (
    "You analyse symptoms and provide a possible diagnosis. Summarise key points in one response."
)
SEED = "I am feeling persistent headaches and fatigue. Can you help?"

CHANNEL_TIMEOUT_S = 120.0

# Synthetic figures for --offline. Chosen to be obviously fake so no one
# mistakes an offline table for a measurement.
_FAKE_USAGE = Usage(prompt_tokens=111, completion_tokens=11, total_tokens=122)


def _offline_config() -> TestConfig:
    return TestConfig(
        ModelResponse(ModelMessage("Offline stub reply."), usage=_FAKE_USAGE),
        provider=ModelProvider.ANTHROPIC,
        model=MODEL,
    )


def _build_agent(config: ModelConfig, ledger: TokenLedger, *, with_tool: bool) -> Agent:
    return Agent(
        AGENT_NAME,
        ROLE,
        config=config,
        tools=[record_consultation_status] if with_tool else [],
        middleware=[ledger.counter_for(AGENT_NAME)],
    )


async def run_bare(config: ModelConfig) -> TokenLedger:
    """Plain model turn. No hub, no plugin, no network tools."""
    ledger = TokenLedger()
    agent = _build_agent(config, ledger, with_tool=False)
    reply = await agent.ask(SEED)
    body = reply.body
    print(f"[bare] reply: {(body or '')[:80]!r}")
    return ledger


async def run_registered(
    config: ModelConfig, *, with_tool: bool
) -> tuple[TokenLedger, list[Envelope]]:
    """Same agent, driven from inside a workflow channel."""
    ledger = TokenLedger()
    agent = _build_agent(config, ledger, with_tool=with_tool)

    # `Hub.open` is L39's entry point and registers the default adapters
    # (consulting, conversation, discussion, workflow). The `Hub(store)`
    # constructor does not — `create_channel(manifest_type=WORKFLOW_TYPE)`
    # then raises NotFoundError until `hub.register_adapter(WorkflowAdapter())`.
    # Both are correct; only one is the repo idiom.
    hub = await Hub.open(MemoryKnowledgeStore())
    hub_client = await HubClient(LocalLink(hub)).open()
    try:
        human = await hub_client.register_human(Passport(name=HUMAN_NAME, kind="human"))
        agent_client = await hub_client.register(agent)

        # initial_speaker and targets are resolved against participant
        # agent_ids, not passport names — workflow.initial_state builds
        # `order` from `p.agent_id`.
        graph = TransitionGraph.sequence([human.agent_id, agent_client.agent_id])

        metadata = await hub_client.create_channel(
            creator_id=human.agent_id,
            manifest_type=WORKFLOW_TYPE,
            participants=[human.agent_id, agent_client.agent_id],
            knobs={"graph": graph.to_dict()},
        )

        closed = asyncio.create_task(
            human.wait_for_channel_event(
                channel_id=metadata.channel_id,
                predicate=lambda env: env.event_type == EV_CHANNEL_CLOSED,
                timeout=CHANNEL_TIMEOUT_S,
            )
        )
        await human.send(metadata.channel_id, SEED)
        await closed

        wal = await hub_client.read_wal(metadata.channel_id)
        return ledger, wal
    finally:
        await hub_client.close()
        await hub.close()


def _print_wal(wal: list[Envelope]) -> None:
    print(f"\nWAL — {len(wal)} envelopes")
    for env in wal:
        print(f"  {env.event_type}")


_A = "A bare"
_B = "B registered"
_C = "C registered + tool"


def _row(label: str, tokens: AgentTokens) -> str:
    return (
        f"{label:<22}{tokens.calls:>7}{tokens.prompt_tokens:>10}"
        f"{tokens.completion_tokens:>13}{tokens.cache_read_input_tokens:>13}"
        f"{tokens.cache_creation_input_tokens:>13}"
    )


def _print_table(rows: list[tuple[str, list[AgentTokens]]], *, offline: bool) -> None:
    print("\n" + "=" * 78)
    if offline:
        print("OFFLINE — token values are synthetic. Call counts are real.")
    header = (
        f"{'run':<22}{'calls':>7}{'prompt':>10}"
        f"{'completion':>13}{'cache_read':>13}{'cache_write':>13}"
    )
    print(header)
    print("-" * 78)
    for label, calls in rows:
        total = AgentTokens()
        for one in calls:
            total = total.plus_tokens(one)
        print(_row(label, total))
        # Summed prompt tokens across a tool loop are not a baseline.
        if len(calls) > 1:
            for i, one in enumerate(calls, start=1):
                print(_row(f"    call {i}", one))
    print("-" * 78)

    by_label = dict(rows)
    if any(not calls for calls in by_label.values()):
        empty = [label for label, calls in rows if not calls]
        print(f"No calls booked for: {empty}.")
        if _B in empty:
            print("Middleware did NOT reach the plugin glue. The source read is wrong.")
        return

    # First-call prompt tokens are the only comparable baseline when runs
    # differ in call count.
    bare = by_label[_A][0].prompt_tokens
    registered = by_label[_B][0].prompt_tokens
    with_tool = by_label[_C][0].prompt_tokens

    registration_delta = registered - bare
    marginal_tool_delta = with_tool - registered

    print("Comparing FIRST CALL of each run (identical seed, identical position).")
    print(f"Registration cost   (B1 - A1): {registration_delta:+,} prompt tokens")
    print(f"Marginal tool cost  (C1 - B1): {marginal_tool_delta:+,} prompt tokens")

    loops = {label: len(calls) for label, calls in rows if len(calls) > 1}
    if loops:
        print(f"\nTool loops observed: {loops}")
        print("Registration grants delegate/peers/channels/tasks/context — a")
        print("registered agent can spend several model calls per turn.")

    if offline:
        print("\nBoth predictions untested — offline tokens are fabricated.")
        return

    if 3000 <= registration_delta <= 5000:
        print("  → inside the predicted 3,000-5,000 registration band.")
    else:
        print(f"  → OUTSIDE the predicted band. Record why: {registration_delta:+,}.")

    if marginal_tool_delta < 1273:
        print("  → under F-L39-10's 1,273 bare-agent tool figure, as predicted.")
    else:
        print("  → at or above 1,273; tool cost is additive, independent of baseline.")

    if registration_delta > marginal_tool_delta:
        print("\nRegistration dominates marginal tool cost.")
    else:
        print("\nMarginal tool cost dominates registration.")


async def main(offline: bool) -> int:
    def _config() -> ModelConfig:
        return _offline_config() if offline else anthropic_config()

    # TestConfig holds a single scripted response and `copy()` returns
    # self, so each offline run needs its own instance.
    bare_ledger = await run_bare(_config())
    plain_ledger, wal = await run_registered(_config(), with_tool=False)
    tooled_ledger, _ = await run_registered(_config(), with_tool=True)

    _print_wal(wal)
    _print_table(
        [
            (_A, bare_ledger.per_call(AGENT_NAME)),
            (_B, plain_ledger.per_call(AGENT_NAME)),
            (_C, tooled_ledger.per_call(AGENT_NAME)),
        ],
        offline=offline,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="exercise the invocation path with a scripted client and no provider",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.offline)))
