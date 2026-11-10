"""Test the OpenRouter Speech TTS entity."""

from collections.abc import AsyncGenerator
import io
from typing import Self
from unittest.mock import MagicMock
import wave

from openai import OpenAIError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.openrouter_speech.tts import pcm_rate, wav_header
from homeassistant.components.tts import (
    ATTR_VOICE,
    TTSAudioRequest,
)
from homeassistant.components.tts.const import DATA_TTS_MANAGER
from homeassistant.components.tts.helper import get_engine_instance
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

ENTITY_ID = "tts.elevenlabs_eleven_v4"


class _AudioStream:
    """Stand-in for the SDK's streamed binary response."""

    def __init__(
        self, chunks: list[bytes] | None = None, content_type: str = "audio/pcm"
    ) -> None:
        self._chunks = chunks
        self.headers = {"content-type": content_type}

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
    ("content_type", "rate"),
    [
        ("audio/pcm", 24000),
        ("audio/pcm; rate=16000", 16000),
        ("audio/pcm;sample_rate=48000", 48000),
        ("audio/pcm; rate=abc", 24000),
        ("", 24000),
    ],
)
def test_pcm_rate(content_type: str, rate: int) -> None:
    """The rate comes from Content-Type when present, else the 24 kHz default."""
    assert pcm_rate(content_type) == rate


def test_wav_header_is_readable() -> None:
    """The streaming header plus PCM parses as 16-bit mono WAV at the given rate."""
    with wave.open(io.BytesIO(wav_header(16000) + b"\x01\x00\x02\x00")) as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(2) == b"\x01\x00\x02\x00"


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
    create.return_value = _AudioStream(
        [b"\x01\x00", b"\x02\x00"], "audio/pcm; rate=16000"
    )
    entity = get_engine_instance(hass, ENTITY_ID)

    response = await entity.async_stream_tts_audio(_request({ATTR_VOICE: "george"}))

    assert response.extension == "wav"
    data = b"".join([c async for c in response.data_gen])
    assert data == wav_header(16000) + b"\x01\x00\x02\x00"
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "elevenlabs/eleven-v4"
    assert kwargs["voice"] == "george"
    assert kwargs["input"] == "Hello, café"
    assert kwargs["response_format"] == "pcm"
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

    assert response.extension == "wav"
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


async def test_speech_manager_stream(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, openai_client: MagicMock
) -> None:
    """The full HA TTS path (as used by Assist) accepts the engine and streams audio."""
    create = openai_client.audio.speech.with_streaming_response.create
    create.return_value = _AudioStream([b"\x01\x00"])

    # Without the file cache, so a cached result cannot hide an engine failure.
    stream = hass.data[DATA_TTS_MANAGER].async_create_result_stream(
        ENTITY_ID,
        use_file_cache=False,
        # A WAV request needs no ffmpeg, which the test environment lacks.
        options={"preferred_format": "wav"},
    )
    stream.async_set_message("Speech manager regression")

    data = b"".join([c async for c in stream.async_stream_result()])
    assert data == wav_header(24000) + b"\x01\x00"
    assert hass.states.get(ENTITY_ID).attributes["friendly_name"] == (
        "ElevenLabs: Eleven v4"
    )
