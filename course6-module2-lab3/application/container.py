from enum import Enum

from dotenv import load_dotenv

from application.interfaces.video_agent_interface import VideoAgentInterface
from application.manual_loop_agent import ManualLoopAgent
from application.recursive_agent import RecursiveAgent
from application.tools.extract_video_id import make_extract_video_id
from application.tools.fetch_transcript import make_fetch_transcript
from application.tools.get_full_metadata import make_get_full_metadata
from application.tools.get_thumbnails import make_get_thumbnails
from application.tools.get_trending_videos import make_get_trending_videos
from application.tools.search_youtube import make_search_youtube
from application.two_step_chain_agent import TwoStepChainAgent
from infra.openai_chat_model import OpenAIChatModelProvider
from infra.pytubefix_search_client import PytubefixSearchClient
from infra.youtube_transcript_api_client import YouTubeTranscriptApiClient
from infra.yt_dlp_metadata_client import YtDlpMetadataClient
from interfaces.chat_model_provider_interface import ChatModelProviderInterface


class AgentStrategy(Enum):
    """Orchestration strategy for the video agent."""

    MANUAL_LOOP = "manual_loop"
    TWO_STEP_CHAIN = "two_step_chain"
    RECURSIVE = "recursive"


def initialise(
    strategy: AgentStrategy = AgentStrategy.RECURSIVE,
    chat_model_provider: ChatModelProviderInterface | None = None,
) -> VideoAgentInterface:
    """Build a fully-wired VideoAgent with the chosen orchestration strategy.

    All three agent implementations share the same tools and the same LLM.
    The only thing that varies is which agent class is instantiated.

    Args:
        strategy: Which orchestration shape to use. Default RECURSIVE.
        chat_model_provider: Optional override for the LLM provider. Tests
            inject a mock here so the OpenAI client is never constructed.

    Returns:
        A VideoAgentInterface ready to receive run(query) calls.
    """
    load_dotenv()

    if chat_model_provider is None:
        chat_model_provider = OpenAIChatModelProvider()
    llm = chat_model_provider.create()

    transcript_client = YouTubeTranscriptApiClient()
    search_client = PytubefixSearchClient()
    metadata_client = YtDlpMetadataClient()

    tools = [
        make_extract_video_id(),
        make_fetch_transcript(transcript_client),
        make_search_youtube(search_client),
        make_get_full_metadata(metadata_client),
        make_get_thumbnails(metadata_client),
        make_get_trending_videos(metadata_client),
    ]

    if strategy is AgentStrategy.MANUAL_LOOP:
        return ManualLoopAgent(llm, tools)
    if strategy is AgentStrategy.TWO_STEP_CHAIN:
        return TwoStepChainAgent(llm, tools)
    if strategy is AgentStrategy.RECURSIVE:
        return RecursiveAgent(llm, tools)
    raise ValueError(f"Unknown agent strategy: {strategy}")