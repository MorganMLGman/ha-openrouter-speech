"""Test the OpenRouter Speech integration module."""

import subprocess
import sys

# Run in a fresh interpreter: the test session has already imported everything.
CHECK_AUDIO_IMPORTS = """
import importlib

import custom_components.openrouter_speech
import openai

imports = []
import_module = importlib.import_module


def record_import(name, package=None):
    imports.append(name)
    return import_module(name, package)


importlib.import_module = record_import
client = openai.AsyncOpenAI(api_key="test")
client.audio.speech
client.audio.transcriptions
print(imports)
"""


def test_audio_resources_are_imported_with_the_integration() -> None:
    """Using client.audio must not import modules inside HA's event loop."""
    result = subprocess.run(
        [sys.executable, "-c", CHECK_AUDIO_IMPORTS],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.strip() == "[]"
