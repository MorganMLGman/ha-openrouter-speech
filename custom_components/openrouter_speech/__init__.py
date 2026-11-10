"""The OpenRouter Speech integration."""

from openai import AsyncOpenAI

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError, ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.httpx_client import get_async_client

from .api import OpenRouterApiError, OpenRouterAuthError, async_get_key_label
from .const import BASE_URL, DOMAIN

PLATFORMS = [Platform.STT, Platform.TTS]

type OpenRouterSpeechConfigEntry = ConfigEntry[AsyncOpenAI]


async def async_setup_entry(
    hass: HomeAssistant, entry: OpenRouterSpeechConfigEntry
) -> bool:
    """Set up OpenRouter Speech from a config entry."""
    try:
        await async_get_key_label(
            async_get_clientsession(hass), entry.data[CONF_API_KEY]
        )
    except OpenRouterAuthError as err:
        raise ConfigEntryError(
            translation_domain=DOMAIN, translation_key="invalid_api_key"
        ) from err
    except OpenRouterApiError as err:
        raise ConfigEntryNotReady(
            translation_domain=DOMAIN,
            translation_key="api_error",
            translation_placeholders={"error": str(err)},
        ) from err

    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )
    # The SDK reads platform info from disk on first use; do it off the event loop.
    await hass.async_add_executor_job(client.platform_headers)
    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(
    hass: HomeAssistant, entry: OpenRouterSpeechConfigEntry
) -> None:
    """Reload so added or removed subentries get their entities."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(
    hass: HomeAssistant, entry: OpenRouterSpeechConfigEntry
) -> bool:
    """Unload OpenRouter Speech."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
