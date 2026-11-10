"""Speech-to-text support for OpenRouter Speech."""

from collections.abc import AsyncIterable
import io
from typing import override
import wave

from openai import OpenAIError

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.core import HomeAssistant
from homeassistant.generated.languages import LANGUAGES
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OpenRouterSpeechConfigEntry
from .const import LOGGER, REQUEST_HEADERS, SUBENTRY_STT
from .entity import OpenRouterSpeechEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenRouterSpeechConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up STT entities."""
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type != SUBENTRY_STT:
            continue
        async_add_entities(
            [OpenRouterSTTEntity(config_entry, subentry)],
            config_subentry_id=subentry_id,
        )


def pcm_to_wav(audio: bytes, metadata: SpeechMetadata) -> bytes:
    """Wrap raw PCM from the Assist pipeline in a WAV header."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(metadata.channel.value)
        wav_file.setsampwidth(metadata.bit_rate.value // 8)
        wav_file.setframerate(metadata.sample_rate.value)
        wav_file.writeframes(audio)
    return buffer.getvalue()


class OpenRouterSTTEntity(SpeechToTextEntity, OpenRouterSpeechEntity):
    """OpenRouter STT entity."""

    @property
    @override
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        # The models API does not list languages per model, so every language is
        # offered and the provider decides.
        return sorted(LANGUAGES)

    @property
    @override
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    @override
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    @override
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bit rates."""
        return list(AudioBitRates)

    @property
    @override
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported sample rates."""
        return list(AudioSampleRates)

    @property
    @override
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return list(AudioChannels)

    @override
    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Send the whole utterance to OpenRouter; the API has no streaming input."""
        audio = b"".join([chunk async for chunk in stream])
        if metadata.format == AudioFormats.WAV:
            audio = pcm_to_wav(audio, metadata)

        try:
            response = await self.entry.runtime_data.audio.transcriptions.create(
                model=self.model,
                file=(f"audio.{metadata.format.value}", audio),
                response_format="json",
                language=metadata.language.split("-")[0],
                extra_headers=REQUEST_HEADERS,
            )
        except OpenAIError:
            LOGGER.exception("Error during speech-to-text")
            return SpeechResult(None, SpeechResultState.ERROR)

        if not response.text:
            return SpeechResult(None, SpeechResultState.ERROR)
        return SpeechResult(response.text, SpeechResultState.SUCCESS)
