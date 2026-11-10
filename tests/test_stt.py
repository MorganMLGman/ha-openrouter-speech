"""Test the OpenRouter Speech STT entity."""

from collections.abc import AsyncIterable
import io
from unittest.mock import MagicMock
import wave

from openai import OpenAIError
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.typing import ClientSessionGenerator

from homeassistant.components import stt
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResultState,
)
from homeassistant.core import HomeAssistant

ENTITY_ID = "stt.openai_whisper_1"


def _metadata(audio_format: AudioFormats = AudioFormats.WAV) -> SpeechMetadata:
    return SpeechMetadata(
        language="pl-PL",
        format=audio_format,
        codec=AudioCodecs.PCM,
        bit_rate=AudioBitRates.BITRATE_16,
        sample_rate=AudioSampleRates.SAMPLERATE_16000,
        channel=AudioChannels.CHANNEL_MONO,
    )


async def _stream(*chunks: bytes) -> AsyncIterable[bytes]:
    for chunk in chunks:
        yield chunk


async def test_transcribe_wraps_pcm_in_wav(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, openai_client: MagicMock
) -> None:
    """Raw pipeline PCM is sent as a valid WAV with the language region stripped."""
    openai_client.audio.transcriptions.create.return_value = MagicMock(
        text="turn on the café lights"
    )
    entity = stt.async_get_speech_to_text_entity(hass, ENTITY_ID)

    result = await entity.async_process_audio_stream(
        _metadata(), _stream(b"\x01\x00" * 4, b"\x02\x00" * 4)
    )

    assert result.result is SpeechResultState.SUCCESS
    assert result.text == "turn on the café lights"
    kwargs = openai_client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "openai/whisper-1"
    assert kwargs["language"] == "pl"
    name, payload = kwargs["file"]
    assert name == "audio.wav"
    with wave.open(io.BytesIO(payload)) as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(16) == b"\x01\x00" * 4 + b"\x02\x00" * 4


async def test_transcribe_ogg_passthrough(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, openai_client: MagicMock
) -> None:
    """OGG audio is already a container and is sent unchanged."""
    openai_client.audio.transcriptions.create.return_value = MagicMock(text="ok")
    entity = stt.async_get_speech_to_text_entity(hass, ENTITY_ID)

    await entity.async_process_audio_stream(
        _metadata(AudioFormats.OGG), _stream(b"OggS-data")
    )

    assert openai_client.audio.transcriptions.create.call_args.kwargs["file"] == (
        "audio.ogg",
        b"OggS-data",
    )


async def test_transcribe_api_error(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, openai_client: MagicMock
) -> None:
    """An API failure becomes an STT error result, not an exception."""
    openai_client.audio.transcriptions.create.side_effect = OpenAIError("boom")
    entity = stt.async_get_speech_to_text_entity(hass, ENTITY_ID)

    result = await entity.async_process_audio_stream(_metadata(), _stream(b""))

    assert result.result is SpeechResultState.ERROR


async def test_transcribe_empty_text(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, openai_client: MagicMock
) -> None:
    """Silence transcribed to an empty string is an error, not an empty command."""
    openai_client.audio.transcriptions.create.return_value = MagicMock(text="")
    entity = stt.async_get_speech_to_text_entity(hass, ENTITY_ID)

    result = await entity.async_process_audio_stream(_metadata(), _stream(b"\x00"))

    assert result.result is SpeechResultState.ERROR


async def test_stt_http_endpoint(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    openai_client: MagicMock,
    hass_client: ClientSessionGenerator,
) -> None:
    """HA's own STT path (metadata checks, then the entity) accepts pipeline audio."""
    openai_client.audio.transcriptions.create.return_value = MagicMock(
        text="turn on the lights"
    )
    client = await hass_client()

    response = await client.post(
        f"/api/stt/{ENTITY_ID}",
        headers={
            "X-Speech-Content": (
                "format=wav; codec=pcm; sample_rate=16000; bit_rate=16; "
                "channel=1; language=pl"
            )
        },
        data=b"\x00\x00" * 160,
    )

    assert response.status == 200
    assert await response.json() == {"text": "turn on the lights", "result": "success"}
    assert (
        openai_client.audio.transcriptions.create.call_args.kwargs["language"] == "pl"
    )
