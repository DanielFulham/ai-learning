"""Notebook cell 20: multi-agent orchestration.

Classic (what the notebook ships):
    groupchat = GroupChat(agents=[teacher, planner, reviewer],
                          speaker_selection_method="auto")
    manager   = GroupChatManager(name="group_manager", groupchat=groupchat,
                                 llm_config=llm_config)
    teacher.initiate_chat(recipient=manager, message="Make a simple lesson about the moon.",
                          max_turns=6)

All agents share one in-process message list; a GroupChatManager LLM picks
who speaks next each round; termination is a magic string ("DONE") plus a
turn cap. The whole thing is a for-loop over a Python list.

v1.0 (here): a Hub, registered agents, and a declarative TransitionGraph
driving a workflow channel.

    hub     = await Hub.open(MemoryKnowledgeStore())
    graph   = TransitionGraph.sequence([requester_id, planner_id, reviewer_id])
    channel = await requester.open(type="workflow",
                                   target=[planner_id, reviewer_id],
                                   knobs={"graph": graph.to_dict()})
    await channel.send("Make a simple lesson about the moon.")

The notebook's third role, the teacher, becomes a HUMAN participant.
Getting there took three wrong turns, and they are the finding:

  1. The opening seat's turn IS the message you send. ``initial_state``
     sets ``expected_next_speaker = graph.initial_speaker``, and the
     seed envelope is folded as that speaker's turn. So the initial
     speaker never runs its own model - a teacher agent in that seat is
     dead configuration.

  2. Generating the seed separately (``teacher_agent.ask(TOPIC)``)
     silently opened a SECOND channel. ``hub.register(agent)`` installs
     the network plugin, which grants delegation tools, so ``ask()`` on
     a registered agent is no longer a plain model turn - the teacher
     consulted the planner over a consulting channel, got a lesson plan
     back, and returned a summary of it. Five model calls instead of
     three, and that exchange lives in a different channel's WAL.

  3. Dropping to ``sequence([planner, reviewer])`` left only ONE
     thinking agent, since the planner then held the opening seat. The
     reviewer received a bare topic and wrote the plan itself. An
     N-step sequence runs N-1 agents.

The seat is real and necessary; it just belongs to whoever supplies the
seed. Here that is a person, so ``register_human`` is the honest way to
say it - no Agent attached, no plugin, therefore no delegation and no
phantom channel. Classic had no non-LLM participant except
UserProxyAgent, which is why the notebook dressed a human up as a
teacher.

Three differences that matter beyond the vocabulary:

  1. Routing is DATA, not an LLM decision. ``speaker_selection_method="auto"``
     asks a manager model who speaks next, so the topology only exists at
     runtime and can differ between identical runs. A TransitionGraph is a
     JSON-serialisable object you can print, diff and store. AG2 v1.0 still
     offers LLM-chosen ordering (the ``discussion`` adapter), but the
     workflow adapter makes declared routing the default shape.

  2. The channel has an append-only log. ``hub.read_wal(channel_id)``
     returns every envelope the channel carried, so the run is replayable
     and auditable after the fact. Classic's message list died with the
     process.

  3. No orchestrator LLM. GroupChatManager burns a model call per round
     just to choose a speaker. ``sequence`` costs zero - the transition is
     evaluated in Python.

``sequence`` sets ``max_turns=len(steps)`` and terminates with
``reason="sequence_complete"``, so the notebook's "reply DONE when
satisfied" magic string is unnecessary; termination is structural.

Registered agents get ``attach_plugin=True`` by default, which wires the
receive -> run-the-agent -> reply glue. There is no message-handling code
in this file for that reason.
"""

from __future__ import annotations

import asyncio

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    BaseHubListener,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    TransitionGraph,
)

from helpers.config import build_config

TOPIC = "Make a simple lesson about the moon."

PLANNER_PROMPT = (
    "You are a classroom lesson planner. Given a topic, write a short lesson "
    "plan for a fourth grade class."
)
REVIEWER_PROMPT = (
    "You are a classroom lesson reviewer. Compare the plan to the curriculum "
    "and suggest up to 3 brief improvements."
)


def _is_closed(envelope: Envelope) -> bool:
    return envelope.event_type == EV_CHANNEL_CLOSED


class LiveTrace(BaseHubListener):
    """Print each envelope as the hub accepts it, so the run is visible."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    async def on_envelope_posted(self, envelope: Envelope, metadata: object) -> None:
        sender = self.names.get(envelope.sender_id, envelope.sender_id[:8])
        text = str(envelope.event_data.get("text", envelope.event_data))
        preview = text if len(text) <= 300 else text[:300] + " ..."
        print(f"\n>> [{sender}] {envelope.event_type}\n{preview}", flush=True)

    async def on_envelope_rejected(self, envelope: Envelope, reason: object) -> None:
        print(f"\n!! REJECTED from {envelope.sender_id[:8]}: {reason}", flush=True)

    async def on_channel_event(self, channel_id: str, kind: str, payload: object) -> None:
        print(f"\n== channel {channel_id[:8]} :: {kind} :: {payload}", flush=True)


async def main() -> None:
    config = build_config()

    async with await Hub.open(MemoryKnowledgeStore()) as hub:
        trace = LiveTrace()
        hub.register_listener(trace)

        hub_client = HubClient(LocalLink(hub), hub=hub)
        requester = await hub_client.register_human(Passport(name="requester"))

        planner = await hub.register(Agent("planner", PLANNER_PROMPT, config=config))
        reviewer = await hub.register(Agent("reviewer", REVIEWER_PROMPT, config=config))

        trace.names = {
            requester.agent_id: "requester(human)",
            planner.agent_id: "planner",
            reviewer.agent_id: "reviewer",
        }

        graph = TransitionGraph.sequence(
            [requester.agent_id, planner.agent_id, reviewer.agent_id]
        )
        print(f"Graph (declared, serialisable):\n{graph.dumps()}\n")

        channel = await requester.open(
            type="workflow",
            target=[planner.agent_id, reviewer.agent_id],
            knobs={"graph": graph.to_dict()},
        )
        await channel.send(TOPIC)

        closed = await requester.wait_for_channel_event(
            channel_id=channel.channel_id,
            predicate=_is_closed,
            timeout=180.0,
        )
        print(f"Channel closed: {closed.event_data.get('reason', '(no reason given)')}\n")

        print("--- write-ahead log ---")
        for envelope in await hub.read_wal(channel.channel_id):
            sender = trace.names.get(envelope.sender_id, envelope.sender_id)
            text = str(envelope.event_data.get("text", envelope.event_data))
            print(f"\n[{sender}] ({envelope.event_type})\n{text}")


if __name__ == "__main__":
    asyncio.run(main())
