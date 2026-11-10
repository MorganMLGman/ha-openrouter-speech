"""Base entity for OpenRouter Speech."""

from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity

from . import OpenRouterSpeechConfigEntry
from .const import DOMAIN


class OpenRouterSpeechEntity(Entity):
    """One entity per subentry, each on its own service device."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: OpenRouterSpeechConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.model: str = subentry.data[CONF_MODEL]
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="OpenRouter",
            model=self.model,
            entry_type=dr.DeviceEntryType.SERVICE,
        )
