"""Test the OpenRouter Speech config flow."""

from probatio import UNDEFINED
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.openrouter_speech.const import BASE_URL, DOMAIN
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType


async def test_user_flow(hass: HomeAssistant, api: AiohttpClientMocker) -> None:
    """A valid key creates an entry titled with the key label."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-test"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "home-key"
    assert result["data"] == {CONF_API_KEY: "sk-test"}


async def test_user_flow_invalid_key(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A rejected key shows an error instead of creating an entry."""
    aioclient_mock.get(f"{BASE_URL}/key", status=401)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "bad"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}


async def test_user_flow_cannot_connect(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A server error is reported as a connection problem."""
    aioclient_mock.get(f"{BASE_URL}/key", status=500)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-test"}
    )
    assert result["errors"] == {"base": "cannot_connect"}


async def test_user_flow_duplicate_key(
    hass: HomeAssistant, config_entry: MockConfigEntry, api: AiohttpClientMocker
) -> None:
    """The same key cannot be added twice."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-test"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_stt_subentry(hass: HomeAssistant, loaded_entry: MockConfigEntry) -> None:
    """The STT subentry offers transcription models and stores the pick."""
    result = await hass.config_entries.subentries.async_init(
        (loaded_entry.entry_id, "stt"), context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    options = result["data_schema"].schema[CONF_MODEL].config["options"]
    assert [o["value"] for o in options] == ["openai/whisper-1"]

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_MODEL: "openai/whisper-1"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "OpenAI: Whisper 1"
    assert result["data"] == {CONF_MODEL: "openai/whisper-1"}


async def test_tts_subentry_model_voices(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """The voice step lists the chosen model's own voices."""
    result = await hass.config_entries.subentries.async_init(
        (loaded_entry.entry_id, "tts"), context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_MODEL: "elevenlabs/eleven-v4"}
    )
    assert result["step_id"] == "voice"
    options = result["data_schema"].schema["tts_voice"].config["options"]
    assert [o["value"] for o in options] == ["george", "sarah"]

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"tts_voice": "sarah", "tts_speed": 1.2}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {
        CONF_MODEL: "elevenlabs/eleven-v4",
        "supported_voices": ["george", "sarah"],
        "tts_voice": "sarah",
        "tts_speed": 1.2,
    }


async def test_tts_subentry_typed_model_and_voice(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A model missing from the list and a voice typed by hand are accepted."""
    result = await hass.config_entries.subentries.async_init(
        (loaded_entry.entry_id, "tts"), context={"source": SOURCE_USER}
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_MODEL: "vendor/unlisted-tts"}
    )
    assert result["data_schema"].schema["tts_voice"].config["options"] == []

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"tts_voice": "custom-voice"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "vendor/unlisted-tts"
    assert result["data"]["supported_voices"] == []
    assert result["data"]["tts_voice"] == "custom-voice"


async def test_subentry_no_models(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """An empty catalogue aborts instead of showing an empty dropdown."""
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE_URL}/models", json={"data": []})
    result = await hass.config_entries.subentries.async_init(
        (loaded_entry.entry_id, "stt"), context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_models"


async def test_subentry_cannot_connect(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """A failing models request aborts the subentry flow."""
    aioclient_mock.clear_requests()
    aioclient_mock.get(f"{BASE_URL}/models", status=500)
    result = await hass.config_entries.subentries.async_init(
        (loaded_entry.entry_id, "tts"), context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "cannot_connect"


async def test_stt_reconfigure(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    aioclient_mock: AiohttpClientMocker,
) -> None:
    """Changing the STT model updates the subentry data and title in place."""
    aioclient_mock.clear_requests()
    aioclient_mock.get(
        f"{BASE_URL}/models",
        json={
            "data": [
                {"id": "openai/whisper-1", "name": "OpenAI: Whisper 1"},
                {"id": "mistralai/voxtral-mini-transcribe", "name": "Voxtral"},
            ]
        },
    )
    result = await loaded_entry.start_subentry_reconfigure_flow(hass, "stt_sub")
    assert result["step_id"] == "user"
    schema = result["data_schema"].schema
    assert next(k for k in schema if k == CONF_MODEL).default() == "openai/whisper-1"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_MODEL: "mistralai/voxtral-mini-transcribe"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    subentry = loaded_entry.subentries["stt_sub"]
    assert subentry.title == "Voxtral"
    assert subentry.data == {CONF_MODEL: "mistralai/voxtral-mini-transcribe"}


async def test_tts_reconfigure_keeps_voice_and_speed(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Reconfiguring with the same model prefills the current voice and speed."""
    result = await loaded_entry.start_subentry_reconfigure_flow(hass, "tts_sub")
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_MODEL: "elevenlabs/eleven-v4"}
    )
    schema = result["data_schema"].schema
    assert next(k for k in schema if k == "tts_voice").default() == "sarah"
    assert next(k for k in schema if k == "tts_speed").default() == 1.0

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"tts_voice": "george", "tts_speed": 1.5}
    )
    assert result["reason"] == "reconfigure_successful"
    data = loaded_entry.subentries["tts_sub"].data
    assert data["tts_voice"] == "george"
    assert data["tts_speed"] == 1.5


async def test_tts_reconfigure_new_model_drops_foreign_voice(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A model without the old voice must not keep it as the default."""
    result = await loaded_entry.start_subentry_reconfigure_flow(hass, "tts_sub")
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_MODEL: "fish-audio/s1"}
    )
    schema = result["data_schema"].schema
    assert schema["tts_voice"].config["options"] == []
    assert next(k for k in schema if k == "tts_voice").default is UNDEFINED

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"tts_voice": "fish-voice"}
    )
    assert result["reason"] == "reconfigure_successful"
    subentry = loaded_entry.subentries["tts_sub"]
    assert subentry.title == "Fish Audio: S1"
    assert subentry.data["supported_voices"] == []
    assert subentry.data["tts_voice"] == "fish-voice"


async def test_tts_reconfigure_keeps_typed_voice_for_same_model(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A hand-typed voice survives reconfiguring the same list-less model."""
    hass.config_entries.async_update_subentry(
        loaded_entry,
        loaded_entry.subentries["tts_sub"],
        data={
            CONF_MODEL: "fish-audio/s1",
            "supported_voices": [],
            "tts_voice": "fish-voice",
            "tts_speed": 1.0,
        },
    )
    await hass.async_block_till_done()
    result = await loaded_entry.start_subentry_reconfigure_flow(hass, "tts_sub")
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_MODEL: "fish-audio/s1"}
    )
    schema = result["data_schema"].schema
    assert next(k for k in schema if k == "tts_voice").default() == "fish-voice"


async def test_tts_reconfigure_keeps_typed_voice_for_listed_model(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A voice typed beside the model's list is not replaced by its first voice."""
    hass.config_entries.async_update_subentry(
        loaded_entry,
        loaded_entry.subentries["tts_sub"],
        data={**loaded_entry.subentries["tts_sub"].data, "tts_voice": "callum"},
    )
    await hass.async_block_till_done()
    result = await loaded_entry.start_subentry_reconfigure_flow(hass, "tts_sub")
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_MODEL: "elevenlabs/eleven-v4"}
    )
    schema = result["data_schema"].schema
    assert next(k for k in schema if k == "tts_voice").default() == "callum"
