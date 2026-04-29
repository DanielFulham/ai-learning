"""
Personal Storyteller — local Mistral via Ollama + gTTS.

Usage:
    python storyteller.py "the life cycle of butterflies"
    python storyteller.py  # uses default topic

Requires:
    - Ollama running locally with `mistral` pulled (`ollama pull mistral`)
    - pip install gTTS langchain-ollama
    - Internet access (gTTS hits Google's TTS endpoint)
"""

import sys
from pathlib import Path

from gtts import gTTS
from langchain_ollama import OllamaLLM


# --- Config ---
MODEL_ID = "mistral"
OUTPUT_DIR = Path("output")
DEFAULT_TOPIC = "the life cycle of butterflies"


def generate_story(topic: str, model: OllamaLLM) -> str:
    """Generate a beginner-friendly educational story on the given topic."""
    prompt = f"""Write an engaging and educational story about {topic} for beginners.
            Use simple and clear language to explain basic concepts.
            Include interesting facts and keep it friendly and encouraging.
            The story should be around 200-300 words and end with a brief summary of what we learned.
            Make it perfect for someone just starting to learn about this topic."""
    return model.invoke(prompt)


def text_to_speech(text: str, output_path: Path) -> None:
    """Convert text to MP3 via gTTS and save to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts = gTTS(text)
    tts.save(str(output_path))


def slugify(text: str) -> str:
    """Turn a topic into a safe filename fragment."""
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC

    print(f"Topic: {topic}")
    print("Loading model and generating story...")

    model = OllamaLLM(model=MODEL_ID, temperature=0.0, num_predict=1000)
    story = generate_story(topic, model)

    print("\n--- Generated Story ---\n")
    print(story)
    print("\n-----------------------\n")

    output_path = OUTPUT_DIR / f"story_{slugify(topic)}.mp3"
    print(f"Converting to speech: {output_path}")
    text_to_speech(story, output_path)
    print(f"Saved audio to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()