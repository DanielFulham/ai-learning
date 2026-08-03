"""Same repro, but with consultation_workflow's actual graph shape."""

import asyncio
import json

from ag2 import Agent
from ag2.config import ModelProvider
from ag2.events import ToolCallEvent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    WORKFLOW_TYPE,
    Hub,
    HubClient,
    LocalLink,
    Passport,
)
from ag2.testing import TestConfig

import consultation_workflow as cw
from helpers.routing import record_consultation_status


def _cfg(*events: object) -> TestConfig:
    return TestConfig(*events, provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5")


async def main() -> None:
    diagnosis = Agent("diagnosis", "d", config=_cfg("diagnosis text"))
    pharmacy = Agent("pharmacy", "p", config=_cfg("pharmacy text"))
    consultation = Agent(
        "consultation",
        "c",
        config=_cfg(
            ToolCallEvent(
                name="record_consultation_status",
                arguments=json.dumps({"status": "complete"}),
            ),
            "final summary",
        ),
        tools=[record_consultation_status],
    )

    hub = await Hub.open(MemoryKnowledgeStore())
    hub_client = await HubClient(LocalLink(hub)).open()
    try:
        human = await hub_client.register_human(Passport(name="patient", kind="human"))
        clients = [await hub_client.register(a) for a in (diagnosis, pharmacy, consultation)]
        step_ids = [c.agent_id for c in clients]
        graph = cw._build_graph(human.agent_id, step_ids)
        meta = await hub_client.create_channel(
            creator_id=human.agent_id,
            manifest_type=WORKFLOW_TYPE,
            participants=[human.agent_id, *step_ids],
            knobs={"graph": graph.to_dict()},
        )
        closed = asyncio.create_task(
            human.wait_for_channel_event(
                channel_id=meta.channel_id,
                predicate=lambda env: env.event_type == EV_CHANNEL_CLOSED,
                timeout=25.0,
            )
        )
        await human.send(meta.channel_id, "seed")
        try:
            await closed
            print("CLOSED normally")
        except TimeoutError:
            print("TIMED OUT — reproduced the hang")
        for env in await hub_client.read_wal(meta.channel_id):
            print(" ", env.event_type)
    finally:
        await hub_client.close()
        await hub.close()


if __name__ == "__main__":
    asyncio.run(main())
