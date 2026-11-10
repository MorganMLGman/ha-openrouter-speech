"""Text-to-speech support for OpenRouter Speech."""

from collections.abc import AsyncGenerator, Mapping
import struct
from typing import Any, override

from openai import OpenAIError
from propcache.api import cached_property

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    Voice,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.generated.languages import LANGUAGES
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OpenRouterSpeechConfigEntry
from .const import (
    CONF_SUPPORTED_VOICES,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    DOMAIN,
    LOGGER,
    RECOMMENDED_TTS_SPEED,
    REQUEST_HEADERS,
    SUBENTRY_TTS,
)
from .entity import OpenRouterSpeechEntity

# OpenRouter's speech endpoint accepts only "mp3" and "pcm", and some models
# (Gemini) only "pcm": 16-bit little-endian mono, rate not documented.
# shortcut: 24 kHz is assumed unless Content-Type names a rate; confirm per
# provider if a voice plays too fast or too slow.
DEFAULT_PCM_RATE = 24000


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenRouterSpeechConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up TTS entities."""
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type != SUBENTRY_TTS:
            continue
        async_add_entities(
            [OpenRouterTTSEntity(config_entry, subentry)],
            config_subentry_id=subentry_id,
        )


def pcm_rate(content_type: str) -> int:
    """Read the sample rate from a Content-Type such as "audio/pcm; rate=24000"."""
    for param in content_type.split(";")[1:]:
        key, _, value = param.strip().partition("=")
        if key.lower() in ("rate", "sample_rate", "samplerate") and value.isdigit():
            return int(value)
    return DEFAULT_PCM_RATE


def wav_header(sample_rate: int) -> bytes:
    """WAV header for a 16-bit mono stream of unknown length."""
    # Sizes are set to the maximum because the stream length is unknown up front;
    # ffmpeg reads such a stream until it ends.
    unknown = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", unknown)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", unknown)
    )


class OpenRouterTTSEntity(TextToSpeechEntity, OpenRouterSpeechEntity):
    """OpenRouter TTS entity."""

    # HA's TTS manager refuses an engine whose name is None ("TTS engine name is
    # not set"), so the entity is named after the subentry instead of its device.
    _attr_has_entity_name = False
    # Without ATTR_PREFERRED_FORMAT here, HA converts our WAV to whatever the
    # caller asked for.
    _attr_supported_options = [ATTR_VOICE]
    # The models API does not list languages per model, so every language is
    # offered and the provider decides.
    _attr_supported_languages = sorted(LANGUAGES)
    _attr_default_language = "en"

    def __init__(
        self, entry: OpenRouterSpeechConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        self._attr_name = subentry.title

    @callback
    @override
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return the model's voices plus the configured one, which may be typed."""
        voices = list(self.subentry.data.get(CONF_SUPPORTED_VOICES) or [])
        if (configured := self.subentry.data[CONF_TTS_VOICE]) not in voices:
            voices.insert(0, configured)
        return [Voice(voice, voice) for voice in voices]

    @cached_property
    @override
    def default_options(self) -> Mapping[str, Any]:
        """Return a mapping with the default options."""
        return {ATTR_VOICE: self.subentry.data[CONF_TTS_VOICE]}

    @override
    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Generate speech; the text is buffered, the audio is streamed back."""
        message = "".join([chunk async for chunk in request.message_gen])
        options = {**self.default_options, **request.options}
        client = self.entry.runtime_data

        async def data_gen() -> AsyncGenerator[bytes]:
            try:
                async with client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=options[ATTR_VOICE],
                    input=message,
                    response_format="pcm",
                    speed=self.subentry.data.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED),
                    extra_headers=REQUEST_HEADERS,
                ) as response:
                    content_type = response.headers.get("content-type", "")
                    LOGGER.debug("Speech response Content-Type: %s", content_type)
                    yield wav_header(pcm_rate(content_type))
                    async for chunk in response.iter_bytes():
                        yield chunk
            except OpenAIError as err:
                LOGGER.debug("Error during text-to-speech", exc_info=True)
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="api_error",
                    translation_placeholders={"error": str(err)},
                ) from err

        return TTSAudioResponse("wav", data_gen())
