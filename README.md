# OpenRouter Speech for Home Assistant

A HACS custom integration that adds OpenRouter **speech-to-text** and **text-to-speech** to Home Assistant, so a whole Assist pipeline can run on one OpenRouter API key.

The built-in OpenRouter integration covers conversation and AI tasks only. This one uses its own domain (`openrouter_speech`) and runs alongside it.

## Installation

1. HACS → ⋮ → Custom repositories → add `https://github.com/MorganMLGman/ha-openrouter-speech`, category **Integration**.
2. Install **OpenRouter Speech** and restart Home Assistant.
3. Settings → Devices & services → Add integration → **OpenRouter Speech**, enter an API key.

Requires Home Assistant 2026.10 or newer.

## Usage

On the integration page, use **Add speech-to-text service** or **Add text-to-speech service**. Each one creates a single entity:

- **STT:** pick a transcription model.
- **TTS:** pick a speech model, then a voice and speed. The voice list comes from the model. If the model lists none, type the voice name from the provider's documentation.

The model lists come live from OpenRouter's `/models` endpoint. A model missing from the list can be typed by hand.

Then select the entities in Settings → Voice assistants → your assistant.

## Notes

- Every language is offered, because OpenRouter does not publish languages per model. Check the provider's documentation for what a model actually supports.
- STT sends the whole utterance after you stop speaking, because the transcription endpoint does not accept streamed audio.
- TTS streams the audio back as it is generated.
- To change the model, voice or speed, use ⋮ → **Reconfigure** on the service.

## Credits

The STT and TTS platforms are adapted from two Home Assistant core pull requests: [#177069](https://github.com/home-assistant/core/pull/177069) by @jenrik and [#184941](https://github.com/home-assistant/core/pull/184941) by @joostlek. Both are Apache-2.0.

## Development

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python pytest-homeassistant-custom-component openai ruff mutagen ha-ffmpeg
.venv/bin/ruff check . && .venv/bin/pytest
```
