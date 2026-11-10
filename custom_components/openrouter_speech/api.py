"""Plain REST calls to OpenRouter that the OpenAI SDK does not cover."""

from dataclasses import dataclass
from typing import Any

import aiohttp

from .const import BASE_URL


class OpenRouterAuthError(Exception):
    """The API key was rejected."""


class OpenRouterApiError(Exception):
    """OpenRouter could not be reached or returned an error."""


@dataclass(frozen=True)
class SpeechModel:
    """A model from the OpenRouter catalogue."""

    id: str
    name: str
    voices: list[str]


async def async_get_key_label(session: aiohttp.ClientSession, api_key: str) -> str:
    """Validate the API key and return its label."""
    data = await _async_get(session, "/key", api_key)
    return data["data"].get("label") or "OpenRouter"


async def async_get_models(
    session: aiohttp.ClientSession, api_key: str, output_modality: str
) -> list[SpeechModel]:
    """Return models with the given output modality: "speech" or "transcription"."""
    data = await _async_get(
        session, "/models", api_key, params={"output_modalities": output_modality}
    )
    return [
        SpeechModel(
            id=model["id"],
            name=model.get("name") or model["id"],
            voices=model.get("supported_voices") or [],
        )
        for model in data["data"]
    ]


async def _async_get(
    session: aiohttp.ClientSession,
    path: str,
    api_key: str,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        async with session.get(
            f"{BASE_URL}{path}",
            params=params,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if response.status == 401:
                raise OpenRouterAuthError
            response.raise_for_status()
            return await response.json()
    except (aiohttp.ClientError, TimeoutError) as err:
        raise OpenRouterApiError(str(err)) from err
