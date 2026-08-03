"""Does `Handoff` relay content where static routing does not?

Static routing (`FromSpeaker` -> `AgentTarget`) was measured to fan out:
every agent's turn was caused by the patient's seed envelope, so nobody
saw a predecessor's output. See `_print_inputs` in consultation_workflow.

This runs the same two-step shape with dynamic routing instead: each
agent hands off via a tool returning `Handoff(target=..., reason=...)`.
`_packet_turn_text` and `_packet_text` both have a dedicated
`routing.kind == "handoff"` branch, and `WorkflowAdapter.tools_for`
documents handoff tools as the intended mechanism — so this is the
candidate for "the primitive I should have used".

Reads the causation chain, which is what settles it:
    caused_by = seed        -> fan-out, no relay
    caused_by = prior packet -> relay works, static routing was the wrong pick

Scripted via TestConfig. No provider, no spend.
"""

import asyncio
from collections.abc import Iterable

from ag2 import Agent, tool
from ag2.config import ModelProvider
from ag2.events import ModelResponse, ToolCallEvent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    EV_PACKET,
    EV_TEXT,
    WORKFLOW_TYPE,
    Handoff,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    TransitionGraph,
)
from ag2.testing import TestConfig

SEED = "SEED_SYMPTOMS"


# TestConfig's own parameter type. `object` here is wide enough to accept
# anything at the call sites and too wide to pass through to TestConfig,
# which puts the error in this function rather than where the mistake is.
ScriptedEvent = ModelResponse | ToolCallEvent | Iterable[ToolCallEvent] | str


def _cfg(*events: ScriptedEvent) -> TestConfig:
    return TestConfig(*events, provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5")


@tool
async def hand_to_pharmacy(summary: str) -> Handoff:
    """Pass the case to the pharmacy agent.

    Args:
        summary: What the pharmacy agent needs to know.
    """
    return Handoff(target="pharmacy", reason=summary)


async def main() -> None:
    diagnosis = Agent(
        "diagnosis",
        "d",
        config=_cfg(
            ToolCallEvent(
                name="hand_to_pharmacy",
                arguments='{"summary": "DIAGNOSIS_SAYS_VIRAL"}',
            ),
            "DIAGNOSIS_BODY",
        ),
        tools=[hand_to_pharmacy],
    )
    pharmacy = Agent("pharmacy", "p", config=_cfg("PHARMACY_BODY"))

    hub = await Hub.open(MemoryKnowledgeStore())
    hub_client = await HubClient(LocalLink(hub)).open()
    try:
        human = await hub_client.register_human(Passport(name="patient", kind="human"))
        clients = [
            await hub_client.register(a, attach_plugin=False) for a in (diagnosis, pharmacy)
        ]
        ids = [c.agent_id for c in clients]
        names = {human.agent_id: "patient", ids[0]: "diagnosis", ids[1]: "pharmacy"}

        graph = TransitionGraph.sequence([human.agent_id, *ids])
        meta = await hub_client.create_channel(
            creator_id=human.agent_id,
            manifest_type=WORKFLOW_TYPE,
            participants=[human.agent_id, *ids],
            knobs={"graph": graph.to_dict()},
        )
        closed = asyncio.create_task(
            human.wait_for_channel_event(
                channel_id=meta.channel_id,
                predicate=lambda env: env.event_type == EV_CHANNEL_CLOSED,
                timeout=25.0,
            )
        )
        await human.send(meta.channel_id, SEED)
        try:
            await closed
        except TimeoutError:
            print("TIMED OUT")

        wal = await hub_client.read_wal(meta.channel_id)
        by_id = {e.envelope_id: e for e in wal}
        print("\ncausation chain")
        for env in wal:
            if env.event_type not in (EV_TEXT, EV_PACKET):
                continue
            # causation_id is `str | None` — the seed envelope has no cause,
            # which is the whole point of reading this chain.
            cause = by_id.get(env.causation_id) if env.causation_id else None
            routing = env.event_data.get("routing", {}) or {}
            print(
                f"  {env.event_type:<14} from={names.get(env.sender_id, '?'):<10}"
                f" routing={routing.get('kind', '-'):<8}"
                f" caused_by={cause.event_type if cause else None}"
                f" (from {names.get(cause.sender_id, '?') if cause else '-'})"
            )
            body = env.event_data.get("body")
            if body:
                print(f"       body={body!r}")
    finally:
        await hub_client.close()
        await hub.close()


if __name__ == "__main__":
    asyncio.run(main())
