"""Text-to-speech support for OpenRouter Speech."""

from collections.abc import AsyncGenerator, Mapping
from typing import Any, Literal, cast, override

from openai import OpenAIError
from propcache.api import cached_property

from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    Voice,
)
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

type Codec = Literal["mp3", "opus", "flac", "wav"]
# Only containers ffmpeg can demux: HA feeds our extension to `ffmpeg -f` when it
# converts, and pcm, aac (muxer is adts) and oga (muxer only) break that.
_PASSTHROUGH: tuple[Codec, ...] = ("mp3", "flac", "wav")


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


def resolve_format(preferred: str) -> tuple[str, Codec]:
    """Map the requested format to (container we return, codec OpenRouter sends).

    When the container differs from the request, HA converts it with ffmpeg.
    """
    if preferred in ("ogg", "oga"):
        return "ogg", "opus"
    if preferred in _PASSTHROUGH:
        return preferred, cast(Codec, preferred)
    return "mp3", "mp3"


class OpenRouterTTSEntity(TextToSpeechEntity, OpenRouterSpeechEntity):
    """OpenRouter TTS entity."""

    _attr_supported_options = [ATTR_VOICE, ATTR_PREFERRED_FORMAT]
    # The models API does not list languages per model, so every language is
    # offered and the provider decides.
    _attr_supported_languages = sorted(LANGUAGES)
    _attr_default_language = "en"

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
        # No preferred format here: HA treats any set format as a conversion request.
        return {ATTR_VOICE: self.subentry.data[CONF_TTS_VOICE]}

    @override
    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Generate speech; the text is buffered, the audio is streamed back."""
        message = "".join([chunk async for chunk in request.message_gen])
        options = {**self.default_options, **request.options}
        extension, codec = resolve_format(options.get(ATTR_PREFERRED_FORMAT, "mp3"))
        client = self.entry.runtime_data

        async def data_gen() -> AsyncGenerator[bytes]:
            try:
                async with client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=options[ATTR_VOICE],
                    input=message,
                    response_format=codec,
                    speed=self.subentry.data.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED),
                    extra_headers=REQUEST_HEADERS,
                ) as response:
                    async for chunk in response.iter_bytes():
                        yield chunk
            except OpenAIError as err:
                LOGGER.debug("Error during text-to-speech", exc_info=True)
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="api_error",
                    translation_placeholders={"error": str(err)},
                ) from err

        return TTSAudioResponse(extension, data_gen())
