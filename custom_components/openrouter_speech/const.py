"""Constants for the OpenRouter Speech integration."""

import logging

DOMAIN = "openrouter_speech"
LOGGER = logging.getLogger(__package__)

BASE_URL = "https://openrouter.ai/api/v1"
# OpenRouter app attribution headers.
REQUEST_HEADERS = {
    "X-Title": "Home Assistant OpenRouter Speech",
    "HTTP-Referer": "https://github.com/MorganMLGman/ha-openrouter-speech",
}

SUBENTRY_STT = "stt"
SUBENTRY_TTS = "tts"

CONF_SUPPORTED_VOICES = "supported_voices"
CONF_TTS_SPEED = "tts_speed"
CONF_TTS_VOICE = "tts_voice"

RECOMMENDED_TTS_SPEED = 1.0
