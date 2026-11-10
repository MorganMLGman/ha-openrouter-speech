"""Fixtures for OpenRouter Speech tests."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.openrouter_speech.const import (
    BASE_URL,
    CONF_SUPPORTED_VOICES,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    DOMAIN,
)
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import HomeAssistant

TTS_MODELS = {
    "data": [
        {
            "id": "elevenlabs/eleven-v4",
            "name": "ElevenLabs: Eleven v4",
            "supported_voices": ["george", "sarah"],
        },
        {"id": "fish-audio/s1", "name": "Fish Audio: S1", "supported_voices": []},
    ]
}
STT_MODELS = {
    "data": [{"id": "openai/whisper-1", "name": "OpenAI: Whisper 1"}],
}


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Load integrations from custom_components."""


@pytest.fixture
def api(aioclient_mock: AiohttpClientMocker) -> AiohttpClientMocker:
    """Mock the OpenRouter REST endpoints used outside the OpenAI SDK."""
    aioclient_mock.get(f"{BASE_URL}/key", json={"data": {"label": "home-key"}})
    aioclient_mock.get(
        f"{BASE_URL}/models",
        params={"output_modalities": "speech"},
        json=TTS_MODELS,
    )
    aioclient_mock.get(
        f"{BASE_URL}/models",
        params={"output_modalities": "transcription"},
        json=STT_MODELS,
    )
    return aioclient_mock


@pytest.fixture
def openai_client() -> Generator[MagicMock]:
    """Replace the OpenAI SDK client the integration builds."""
    client = MagicMock()
    client.platform_headers = MagicMock(return_value={})
    client.audio.transcriptions.create = AsyncMock()
    with patch("custom_components.openrouter_speech.AsyncOpenAI", return_value=client):
        yield client


@pytest.fixture
def config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """A config entry with one STT and one TTS subentry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="home-key",
        data={CONF_API_KEY: "sk-test"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: "openai/whisper-1"},
                subentry_id="stt_sub",
                subentry_type="stt",
                title="OpenAI: Whisper 1",
                unique_id=None,
            ),
            ConfigSubentryData(
                data={
                    CONF_MODEL: "elevenlabs/eleven-v4",
                    CONF_SUPPORTED_VOICES: ["george", "sarah"],
                    CONF_TTS_VOICE: "sarah",
                    CONF_TTS_SPEED: 1.0,
                },
                subentry_id="tts_sub",
                subentry_type="tts",
                title="ElevenLabs: Eleven v4",
                unique_id=None,
            ),
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def loaded_entry(
    hass: HomeAssistant,
    config_entry: MockConfigEntry,
    api: AiohttpClientMocker,
    openai_client: MagicMock,
) -> MockConfigEntry:
    """The config entry, set up."""
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return config_entry
