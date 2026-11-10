"""Test the OpenRouter Speech TTS entity."""

from collections.abc import AsyncGenerator
from typing import Self
from unittest.mock import MagicMock

from openai import OpenAIError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.openrouter_speech.tts import resolve_format
from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_VOICE,
    TTSAudioRequest,
)
from homeassistant.components.tts.helper import get_engine_instance
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

ENTITY_ID = "tts.elevenlabs_eleven_v4"


class _AudioStream:
    """Stand-in for the SDK's streamed binary response."""

    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> Self:
        if self._chunks is None:
            raise OpenAIError("boom")
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def iter_bytes(self) -> AsyncGenerator[bytes]:
        for chunk in self._chunks or []:
            yield chunk


async def _text(*parts: str) -> AsyncGenerator[str]:
    for part in parts:
        yield part


def _request(options: dict) -> TTSAudioRequest:
    return TTSAudioRequest(
        language="pl", options=options, message_gen=_text("Hello, ", "café")
    )


@pytest.mark.parametrize(
    ("preferred", "expected"),
    [
        ("mp3", ("mp3", "mp3")),
        ("flac", ("flac", "flac")),
        ("wav", ("wav", "wav")),
        ("ogg", ("ogg", "opus")),
        ("oga", ("ogg", "opus")),
        ("raw", ("mp3", "mp3")),
        ("pcm", ("mp3", "mp3")),
        ("aac", ("mp3", "mp3")),
        ("webm", ("mp3", "mp3")),
    ],
)
def test_resolve_format(preferred: str, expected: tuple[str, str]) -> None:
    """Only containers ffmpeg can read back are returned; HA converts the rest."""
    assert resolve_format(preferred) == expected


async def test_voices_and_defaults(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """The entity exposes the model's voices and the configured default."""
    entity = get_engine_instance(hass, ENTITY_ID)

    assert [v.voice_id for v in entity.async_get_supported_voices("pl")] == [
        "george",
        "sarah",
    ]
    assert entity.default_options == {ATTR_VOICE: "sarah"}


async def test_typed_voice_is_listed(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A voice typed outside the model's list still shows up as supported."""
    hass.config_entries.async_update_subentry(
        loaded_entry,
        loaded_entry.subentries["tts_sub"],
        data={**loaded_entry.subentries["tts_sub"].data, "tts_voice": "callum"},
    )
    await hass.async_block_till_done()
    entity = get_engine_instance(hass, ENTITY_ID)

    assert [v.voice_id for v in entity.async_get_supported_voices("pl")] == [
        "callum",
        "george",
        "sarah",
    ]


async def test_stream_audio(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, openai_client: MagicMock
) -> None:
    """Text is joined, sent with the requested voice, and audio streamed back."""
    create = openai_client.audio.speech.with_streaming_response.create
    create.return_value = _AudioStream([b"ID3", b"audio"])
    entity = get_engine_instance(hass, ENTITY_ID)

    response = await entity.async_stream_tts_audio(
        _request({ATTR_VOICE: "george", ATTR_PREFERRED_FORMAT: "ogg"})
    )

    assert response.extension == "ogg"
    assert b"".join([c async for c in response.data_gen]) == b"ID3audio"
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "elevenlabs/eleven-v4"
    assert kwargs["voice"] == "george"
    assert kwargs["input"] == "Hello, café"
    assert kwargs["response_format"] == "opus"
    assert kwargs["speed"] == 1.0


async def test_stream_audio_uses_default_voice(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, openai_client: MagicMock
) -> None:
    """Without options the configured voice and mp3 are used."""
    create = openai_client.audio.speech.with_streaming_response.create
    create.return_value = _AudioStream([b"x"])
    entity = get_engine_instance(hass, ENTITY_ID)

    response = await entity.async_stream_tts_audio(_request({}))
    _ = [c async for c in response.data_gen]

    assert response.extension == "mp3"
    assert create.call_args.kwargs["voice"] == "sarah"


async def test_stream_audio_api_error(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, openai_client: MagicMock
) -> None:
    """An API failure surfaces as HomeAssistantError while streaming."""
    openai_client.audio.speech.with_streaming_response.create.return_value = (
        _AudioStream(None)
    )
    entity = get_engine_instance(hass, ENTITY_ID)

    response = await entity.async_stream_tts_audio(_request({}))
    with pytest.raises(HomeAssistantError):
        _ = [c async for c in response.data_gen]
