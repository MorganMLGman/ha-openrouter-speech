"""Config flow for OpenRouter Speech."""

from typing import Any, override

import probatio

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .api import (
    OpenRouterApiError,
    OpenRouterAuthError,
    SpeechModel,
    async_get_key_label,
    async_get_models,
)
from .const import (
    CONF_SUPPORTED_VOICES,
    CONF_TTS_SPEED,
    CONF_TTS_VOICE,
    DOMAIN,
    LOGGER,
    RECOMMENDED_TTS_SPEED,
    SUBENTRY_STT,
    SUBENTRY_TTS,
)


def _dropdown(options: list[SelectOptionDict]) -> SelectSelector:
    """Dropdown that also accepts a typed value the API did not return."""
    return SelectSelector(
        SelectSelectorConfig(
            options=options,
            mode=SelectSelectorMode.DROPDOWN,
            custom_value=True,
            sort=True,
        )
    )


class OpenRouterSpeechConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenRouter Speech."""

    VERSION = 1

    @classmethod
    @callback
    @override
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {SUBENTRY_STT: SttFlowHandler, SUBENTRY_TTS: TtsFlowHandler}

    @override
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for the API key."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            try:
                label = await async_get_key_label(
                    async_get_clientsession(self.hass), user_input[CONF_API_KEY]
                )
            except OpenRouterAuthError:
                errors["base"] = "invalid_auth"
            except OpenRouterApiError:
                errors["base"] = "cannot_connect"
            except Exception:
                LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(title=label, data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=probatio.Schema(
                {probatio.Required(probatio.Secret(CONF_API_KEY)): str}
            ),
            errors=errors,
        )


class _ModelFlowHandler(ConfigSubentryFlow):
    """Shared model selection for STT and TTS subentries."""

    output_modality: str

    def __init__(self) -> None:
        """Initialize the subentry flow."""
        self.models: dict[str, SpeechModel] = {}
        self.data: dict[str, Any] = {}
        self.previous_model: str | None = None

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Run the same steps, prefilled with the current settings."""
        self.data = dict(self._get_reconfigure_subentry().data)
        self.previous_model = self.data[CONF_MODEL]
        return await self.async_step_user()

    async def _async_model_step(
        self, user_input: dict[str, Any] | None
    ) -> SubentryFlowResult | None:
        """Show the model form; return None once a model is chosen."""
        if user_input is not None:
            self.data[CONF_MODEL] = user_input[CONF_MODEL]
            return None

        try:
            models = await async_get_models(
                async_get_clientsession(self.hass),
                self._get_entry().data[CONF_API_KEY],
                self.output_modality,
            )
        except OpenRouterApiError, OpenRouterAuthError:
            return self.async_abort(reason="cannot_connect")
        self.models = {model.id: model for model in models}
        if not self.models:
            return self.async_abort(reason="no_models")

        options = [
            SelectOptionDict(value=model.id, label=model.name) for model in models
        ]
        return self.async_show_form(
            step_id="user",
            data_schema=probatio.Schema(
                {
                    probatio.Required(
                        CONF_MODEL, default=self.data.get(CONF_MODEL)
                    ): _dropdown(options)
                }
            ),
        )

    def _title(self) -> str:
        model = self.models.get(self.data[CONF_MODEL])
        return model.name if model else self.data[CONF_MODEL]

    def _async_finish(self) -> SubentryFlowResult:
        if self.source == SOURCE_RECONFIGURE:
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                title=self._title(),
                data=self.data,
            )
        return self.async_create_entry(title=self._title(), data=self.data)


class SttFlowHandler(_ModelFlowHandler):
    """Add a speech-to-text service."""

    output_modality = "transcription"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Pick the transcription model."""
        if form := await self._async_model_step(user_input):
            return form
        return self._async_finish()


class TtsFlowHandler(_ModelFlowHandler):
    """Add a text-to-speech service."""

    output_modality = "speech"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Pick the speech model."""
        if form := await self._async_model_step(user_input):
            return form
        model = self.models.get(self.data[CONF_MODEL])
        self.data[CONF_SUPPORTED_VOICES] = model.voices if model else []
        return await self.async_step_voice()

    async def async_step_voice(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Pick the voice and speed."""
        if user_input is not None:
            self.data.update(user_input)
            return self._async_finish()

        voices: list[str] = self.data[CONF_SUPPORTED_VOICES]
        current = self.data.get(CONF_TTS_VOICE)
        # A typed voice is kept for the same model; a new model starts from its list.
        if current not in voices and self.data[CONF_MODEL] != self.previous_model:
            current = voices[0] if voices else None
        voice_key = (
            probatio.Required(CONF_TTS_VOICE, default=current)
            if current
            else probatio.Required(CONF_TTS_VOICE)
        )
        return self.async_show_form(
            step_id="voice",
            data_schema=probatio.Schema(
                {
                    voice_key: _dropdown(
                        [SelectOptionDict(value=v, label=v) for v in voices]
                    ),
                    probatio.Optional(
                        CONF_TTS_SPEED,
                        default=self.data.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0.25, max=4.0, step=0.05, mode=NumberSelectorMode.BOX
                        )
                    ),
                }
            ),
            description_placeholders={"model": self._title()},
        )
