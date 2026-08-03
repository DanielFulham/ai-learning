"""AutoMed — multi-agent medical consultation. Port of notebook cells 23/25/29.

    Framework exercise, not a medical tool. Model output here is not
    clinical advice.

Structure, and where it departs from the notebook:

* The patient seat is `register_human`, not an `Agent`. The notebook gave
  `patient` an `llm_config` it never used.
* Turn control lives in the graph, not the prompts. The notebook's
  "Only respond once" / "in one response" / "end with CONSULTATION_COMPLETE"
  are all routing instructions written into system messages; the last one had
  no handler at all.
* No `GroupChatManager`. Routing is a declared `TransitionGraph`.
* Every clinical caveat in the prompts below is added. The notebook's entire
  guardrail was a markdown paragraph outside the code.

MEASURED BEHAVIOUR — the agents do not see each other. Every turn is caused
by the patient's seed envelope, so `pharmacy` never receives `diagnosis`'s
output and `consultation` receives neither. Confirmed against both static
routing and `Handoff` (see repro_handoff.py) on ag2==1.0.1 / LocalLink. The
`_print_inputs` section at the end of a run is the standing check on this;
the notebook's premise that "pharmacy follows up on the diagnosis" does not
hold here.

`ConsultationSummary` is rendered into the prompt rather than passed as
`response_schema` — see its docstring.

Usage:
    python consultation_workflow.py
"""

import asyncio
import sys
import time

from ag2 import Agent
from ag2.config import ModelConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    EV_CONTEXT_SET,
    EV_PACKET,
    WORKFLOW_TYPE,
    AgentTarget,
    BaseHubListener,
    ChannelMetadata,
    ContextEquals,
    Envelope,
    FromSpeaker,
    Hub,
    HubClient,
    LocalLink,
    NetworkError,
    Passport,
    TerminateTarget,
    Transition,
    TransitionGraph,
)
from pydantic import BaseModel, Field

from helpers.config import anthropic_config
from helpers.routing import (
    CONSULTATION_STATUS_KEY,
    STATUS_COMPLETE,
    STATUS_ESCALATE,
    record_consultation_status,
)
from helpers.usage import TokenLedger

HUMAN_NAME = "patient"
DIAGNOSIS = "diagnosis"
PHARMACY = "pharmacy"
CONSULTATION = "consultation"

CHANNEL_TIMEOUT_S = 180.0

# Seed + three agent turns. A backstop only — the ContextEquals exit is
# what should end the run. If a session ever terminates on max_turns,
# consultation failed to call the tool and that is worth knowing.
MAX_TURNS = 4

DIAGNOSIS_PROMPT = (
    "You are part of a medical information demo, not a clinical service. "
    "Read the patient's described symptoms and set out possible explanations. "
    "State plainly that this is not a diagnosis and that you are not a clinician. "
    "Keep it under 150 words."
)

PHARMACY_PROMPT = (
    "You suggest over-the-counter options relevant to the symptoms discussed. "
    "Never suggest prescription-only medicines. Always say that a pharmacist or "
    "doctor should confirm anything before it is taken. Do not give dosages. "
    "Keep it under 150 words."
)

CONSULTATION_PROMPT = (
    "You decide whether the patient should see a doctor, and how urgently.\n\n"
    "You are the FINAL step in a fixed pipeline. There are no further turns and "
    "nobody can reply to you. Do not ask the patient questions — no one will "
    "answer them. Decide with the information you have, and if it is thin, say "
    "so and let that push your recommendation towards in-person care.\n\n"
    "If any symptom could indicate an emergency, say so first and advise "
    "immediate in-person care. Summarise the consultation and give clear next steps.\n\n"
    "Call record_consultation_status EXACTLY ONCE, with "
    f"{STATUS_ESCALATE!r} if in-person care is needed, otherwise "
    f"{STATUS_COMPLETE!r}. Then write your final message to the patient. "
    "Calling the tool does not end the consultation on its own — your final "
    "message does."
)


class ConsultationSummary(BaseModel):
    """Shape of the final consultation output. Rendered into the prompt.

    Not usable as `response_schema` on a registered agent: `Agent` is
    `Generic[TResult]` defaulting to `str`, `HubClient.register` takes a bare
    `Agent`, and `TResult` is invariant. The runtime agrees independently —
    `build_round_envelope` encodes `reply.body`, so only text reaches the
    channel. Prompt-embedding is CrewAI's `output_pydantic` mechanism.

    Field descriptions are prompt content, not documentation.
    """

    seek_professional_care: bool = Field(
        description="True if the patient should see a clinician in person."
    )
    urgency: str = Field(
        description="One of: emergency, same-day, routine, none."
    )
    red_flags: list[str] = Field(
        description="Symptoms described that warrant urgent attention. Empty if none."
    )
    next_steps: str = Field(
        description="What the patient should do next, in plain language."
    )


def _render_output_shape(model: type[BaseModel]) -> str:
    """Render the model as prompt text so the schema stays one source of truth."""
    lines = [f"- {name}: {field.description}" for name, field in model.model_fields.items()]
    return "Structure your final message under these headings:\n" + "\n".join(lines)


class LiveTracer(BaseHubListener):
    """Print envelopes as the hub accepts them.

    `on_turn_failed` is the one that matters: the notify handler traps
    exceptions from `agent.ask` and posts no reply, so a failed turn leaves
    the channel open and silent until the 600s auto-close. Without a
    listener there is no signal at all.

    `HubListener` is a Protocol — every method must be `async` and parameter
    names are load-bearing.
    """

    def __init__(self) -> None:
        self._start = time.monotonic()
        # agent_ids share a hub-scoped prefix, so sender_id[:8] is identical
        # for every participant and tells you nothing. Populated after
        # registration.
        self.names: dict[str, str] = {}

    def _stamp(self) -> str:
        return f"[{time.monotonic() - self._start:6.1f}s]"

    def _who(self, agent_id: str) -> str:
        return self.names.get(agent_id, agent_id[-6:])

    async def on_envelope_posted(
        self, envelope: Envelope, metadata: ChannelMetadata
    ) -> None:
        print(f"{self._stamp()} {envelope.event_type:<24} from={self._who(envelope.sender_id)}")

    async def on_envelope_rejected(
        self, envelope: Envelope, reason: NetworkError
    ) -> None:
        print(f"{self._stamp()} REJECTED {envelope.event_type}: {reason}")

    async def on_turn_failed(
        self, channel_id: str, agent_id: str, envelope_id: str, exc: BaseException
    ) -> None:
        print(f"{self._stamp()} TURN FAILED agent={self._who(agent_id)}: {exc!r}")

    async def on_dispatch_failed(
        self, envelope: Envelope, recipient_id: str, reason: BaseException
    ) -> None:
        print(f"{self._stamp()} DISPATCH FAILED to={self._who(recipient_id)}: {reason!r}")


def _build_agents(
    config: ModelConfig, ledger: TokenLedger
) -> tuple[Agent[str], Agent[str], Agent[str]]:
    """Three agents. Only `consultation` holds the routing tool — see helpers/routing.py.

    All three are `Agent[str]`: `response_schema` would change the generic
    parameter and `HubClient.register` only accepts `Agent[str]`. See
    `ConsultationSummary` for why the schema is rendered into the prompt
    instead.
    """
    return (
        Agent(
            DIAGNOSIS,
            DIAGNOSIS_PROMPT,
            config=config,
            middleware=[ledger.counter_for(DIAGNOSIS)],
        ),
        Agent(
            PHARMACY,
            PHARMACY_PROMPT,
            config=config,
            middleware=[ledger.counter_for(PHARMACY)],
        ),
        Agent(
            CONSULTATION,
            f"{CONSULTATION_PROMPT}\n\n{_render_output_shape(ConsultationSummary)}",
            config=config,
            tools=[record_consultation_status],
            middleware=[ledger.counter_for(CONSULTATION)],
        ),
    )


def _build_graph(seed_id: str, step_ids: list[str]) -> TransitionGraph:
    """Declared routing: seed → diagnosis → pharmacy → consultation, plus an exit.

    `_select` sorts transitions by `priority`, so the exits sit at 0 and
    pre-empt the pipeline steps at 1; otherwise consultation's `FromSpeaker`
    transition wins and the graph falls through to `default_target`.

    Two exits rather than one because `ContextEquals` tests equality and a
    missing key reads as None — matching on None would fire on turn one.
    """
    order = [seed_id, *step_ids]
    transitions = [
        Transition(
            when=ContextEquals(key=CONSULTATION_STATUS_KEY, value=STATUS_COMPLETE),
            then=TerminateTarget(reason="consultation_complete"),
            priority=0,
        ),
        Transition(
            when=ContextEquals(key=CONSULTATION_STATUS_KEY, value=STATUS_ESCALATE),
            then=TerminateTarget(reason="escalated_to_clinician"),
            priority=0,
        ),
    ]
    for current, following in zip(order[:-1], order[1:], strict=True):
        transitions.append(
            Transition(
                when=FromSpeaker(current),
                then=AgentTarget(following),
                priority=1,
            )
        )
    return TransitionGraph(
        initial_speaker=seed_id,
        transitions=transitions,
        default_target=TerminateTarget(reason="pipeline_exhausted"),
        max_turns=MAX_TURNS,
    )


async def run_consultation(symptoms: str, config: ModelConfig) -> None:
    ledger = TokenLedger()
    agents = _build_agents(config, ledger)

    # Hub.open registers the default adapters (including workflow). The
    # Hub(store) constructor does not — see F-L40-5.
    hub = await Hub.open(MemoryKnowledgeStore())
    tracer = LiveTracer()
    hub.register_listener(tracer)
    hub_client = await HubClient(LocalLink(hub)).open()
    try:
        human = await hub_client.register_human(Passport(name=HUMAN_NAME, kind="human"))

        # attach_plugin=False is load-bearing. The default grants delegate /
        # peers / channels / tasks / context; `consultation` used `delegate`
        # to open a second channel mid-turn and the workflow stalled. The
        # graph constrains who speaks next in this channel, not whether an
        # agent opens another one. Also ~2,378 prompt tokens per agent.
        clients = [
            await hub_client.register(agent, attach_plugin=False) for agent in agents
        ]
        step_ids = [client.agent_id for client in clients]
        tracer.names = {
            human.agent_id: HUMAN_NAME,
            **{client.agent_id: agent.name for client, agent in zip(clients, agents, strict=True)},
        }

        # initial_speaker and targets resolve against participant agent_ids,
        # not passport names — WorkflowAdapter.initial_state builds `order`
        # from `p.agent_id` and raises ProtocolError on a name.
        graph = _build_graph(human.agent_id, step_ids)

        metadata = await hub_client.create_channel(
            creator_id=human.agent_id,
            manifest_type=WORKFLOW_TYPE,
            participants=[human.agent_id, *step_ids],
            knobs={"graph": graph.to_dict()},
        )

        closed = asyncio.create_task(
            human.wait_for_channel_event(
                channel_id=metadata.channel_id,
                predicate=lambda env: env.event_type == EV_CHANNEL_CLOSED,
                timeout=CHANNEL_TIMEOUT_S,
            )
        )
        print("\nLive trace (elapsed seconds):")
        await human.send(metadata.channel_id, f"I am feeling {symptoms}. Can you help?")
        try:
            await closed
        except TimeoutError:
            print(
                f"\nChannel did not close within {CHANNEL_TIMEOUT_S:.0f}s. "
                "The last traced envelope is where it stopped advancing."
            )

        wal = await hub_client.read_wal(metadata.channel_id)
        _print_transcript(wal, hub.name_to_id_map())
        _print_wal(wal)
        _print_usage(ledger)
        _print_inputs(ledger)
        _check_termination(wal)
    finally:
        await hub_client.close()
        await hub.close()


def _print_transcript(wal: list[Envelope], name_map: dict[str, str]) -> None:
    """Render the run from the WAL rather than from console side effects."""
    id_to_name = {agent_id: name for name, agent_id in name_map.items()}
    print("\n" + "=" * 78)
    print("TRANSCRIPT")
    print("=" * 78)
    for env in wal:
        if env.event_type == EV_PACKET:
            speaker = id_to_name.get(env.sender_id, env.sender_id)
            body = env.event_data.get("body", "")
            print(f"\n--- {speaker} ---\n{body}")
        elif env.event_type == EV_CONTEXT_SET:
            print(f"\n[context set] {env.event_data.get('set', {})}")


def _print_wal(wal: list[Envelope]) -> None:
    print("\n" + "=" * 78)
    print(f"WAL — {len(wal)} envelopes")
    print("=" * 78)
    for env in wal:
        print(f"  {env.event_type}")


def _print_inputs(ledger: TokenLedger) -> None:
    """What each agent was handed, per model call. See the module docstring."""
    print("\n" + "=" * 78)
    print("AGENT INPUTS — what each model call actually received")
    print("=" * 78)
    for name in (DIAGNOSIS, PHARMACY, CONSULTATION):
        for i, digest in enumerate(ledger.inputs(name), start=1):
            print(f"\n{name} call {i}:\n  {digest}")


def _check_termination(wal: list[Envelope]) -> None:
    """Fail loud when the graph closed on the backstop instead of the exit.

    `AnthropicConfig` exposes no `tool_choice` (xai, mistral, zai and openai
    have one), so the routing call cannot be required — only asked for.
    """
    recorded = [env for env in wal if env.event_type == EV_CONTEXT_SET]
    print("\n" + "=" * 78)
    if recorded:
        status = recorded[-1].event_data.get("set", {}).get(CONSULTATION_STATUS_KEY)
        print(f"Terminated on the ContextEquals exit. Recorded status: {status!r}")
        return
    print("NO STATUS RECORDED.")
    print(
        "consultation never called record_consultation_status, so the "
        "ContextEquals exit never fired and the channel closed on the "
        f"max_turns={MAX_TURNS} backstop instead. The routing signal is "
        "prompt-enforced only; treat this run as a failure of that enforcement."
    )


def _print_usage(ledger: TokenLedger) -> None:
    print("\n" + "=" * 78)
    print(f"{'agent':<18}{'calls':>7}{'prompt':>10}{'completion':>13}")
    print("-" * 78)
    for name, tokens in ledger.rows().items():
        print(
            f"{name:<18}{tokens.calls:>7}{tokens.prompt_tokens:>10}"
            f"{tokens.completion_tokens:>13}"
        )
    total = ledger.total()
    print("-" * 78)
    print(
        f"{'TOTAL':<18}{total.calls:>7}{total.prompt_tokens:>10}"
        f"{total.completion_tokens:>13}"
    )


async def main() -> int:
    print("\nAI Healthcare Consultation System — framework demo, not medical advice.")
    symptoms = input("Describe your symptoms: ").strip()
    if not symptoms:
        print("No symptoms entered; nothing to run.")
        return 1
    await run_consultation(symptoms, anthropic_config())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
