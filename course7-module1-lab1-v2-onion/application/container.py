from application.auth_agent_service import AuthAgentService
from application.counter_agent_service import CounterAgentService
from application.graph_builders import (
    build_auth_graph,
    build_counter_graph,
    build_qa_graph,
)
from application.lab_app import LabApp
from application.qa_agent_service import QAAgentService
from application.scripted_input_provider import ScriptedInputProvider
from infra.console_input_provider import ConsoleInputProvider
from infra.console_stream_consumer import ConsoleStreamConsumer
from infra.ollama_chat_model_provider import OllamaChatModelProvider
from infra.openai_chat_model_provider import OpenAIChatModelProvider
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.input_provider_interface import InputProviderInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


# Demo-specific canned auth responses. Used when the entry point asks the
# container to wire a scripted input provider — see `use_scripted_auth_input`
# below. Sized for the post-fix Failure behaviour (clears both username
# and password), so the second InputNode pass re-prompts for both.
_SCRIPTED_AUTH_RESPONSES = [
    "test_user",         # username, first pass
    "wrong",             # password, first pass — fails
    "test_user",         # username, second pass (Failure cleared both)
    "secure_password",   # password, second pass — succeeds
]


def initialise(
    chat_model_provider: ChatModelProviderInterface | None = None,
    input_provider: InputProviderInterface | None = None,
    stream_consumer: StreamConsumerInterface | None = None,
    use_openai: bool = False,
    use_scripted_auth_input: bool = False,
) -> LabApp:
    """Composition root. Wires concretes to interfaces and returns LabApp.

    Stateless — creates fresh instances every call. The entry point caches
    the result if it wants singleton behaviour.

    Optional injection parameters default to production concretes. Tests
    pass `MagicMock(spec=Interface)` for any dependency they want to control.

    `use_openai` and `use_scripted_auth_input` only apply when the
    corresponding concrete isn't explicitly injected. Explicit injection
    wins; the booleans pick the default when no concrete is passed.
    """
    if chat_model_provider is None:
        if use_openai:
            chat_model_provider = OpenAIChatModelProvider()
        else:
            chat_model_provider = OllamaChatModelProvider()

    if input_provider is None:
        if use_scripted_auth_input:
            input_provider = ScriptedInputProvider(_SCRIPTED_AUTH_RESPONSES)
        else:
            input_provider = ConsoleInputProvider()

    if stream_consumer is None:
        stream_consumer = ConsoleStreamConsumer()

    model = chat_model_provider.create()

    auth_graph = build_auth_graph(input_provider)
    qa_graph = build_qa_graph(model)
    counter_graph = build_counter_graph()

    return LabApp(
        auth=AuthAgentService(auth_graph, stream_consumer),
        qa=QAAgentService(qa_graph, stream_consumer),
        counter=CounterAgentService(counter_graph, stream_consumer),
    )