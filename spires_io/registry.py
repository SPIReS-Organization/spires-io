"""Sensor registry and adapter definitions for SPIReS I/O."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import xarray as xr

import spires_io.modis as modis
import spires_io.viirs as viirs

OpenFn = Callable[..., xr.Dataset]
PrepareFn = Callable[..., xr.Dataset]


@dataclass(frozen=True)
class SensorIOAdapter:
    """Callable I/O adapter surface for a supported sensor family."""

    sensor: str
    supported_platforms: tuple[str, ...]
    open_surface_reflectance: OpenFn
    prepare_scene_for_inversion: PrepareFn
    notes: str = ""


SENSOR_ALIASES = {
    "modis": "modis",
    "terra": "modis",
    "aqua": "modis",
    "viirs": "viirs",
    "snpp": "viirs",
    "npp": "viirs",
    "noaa20": "viirs",
    "j01": "viirs",
    "noaa21": "viirs",
    "j02": "viirs",
}

PLATFORM_ALIASES = {
    "modis": {
        "terra": "terra",
        "mod09ga": "terra",
        "aqua": "aqua",
        "myd09ga": "aqua",
    },
    "viirs": {
        "snpp": "snpp",
        "npp": "snpp",
        "vnp09ga": "snpp",
        "noaa20": "noaa20",
        "j01": "noaa20",
        "vj109ga": "noaa20",
        "noaa21": "noaa21",
        "j02": "noaa21",
        "vj209ga": "noaa21",
    },
}


def _normalize_token(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    return text or None


def normalize_sensor_name(sensor: str) -> str:
    """Return the canonical sensor family name."""
    key = _normalize_token(sensor)
    if key is None:
        raise ValueError("sensor must be provided")
    try:
        return SENSOR_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported sensor {sensor!r}") from exc


def normalize_platform_name(sensor: str, platform: str | None) -> str | None:
    """Return the canonical platform name for a sensor, if provided."""
    if platform is None:
        return None
    canonical_sensor = normalize_sensor_name(sensor)
    key = _normalize_token(platform)
    if key is None:
        return None
    aliases = PLATFORM_ALIASES.get(canonical_sensor, {})
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported platform {platform!r} for sensor {canonical_sensor!r}") from exc


MODIS_ADAPTER = SensorIOAdapter(
    sensor="modis",
    supported_platforms=("terra", "aqua"),
    open_surface_reflectance=modis.open_modis_surface_reflectance,
    prepare_scene_for_inversion=modis.prepare_modis_scene_for_inversion,
    notes="MODIS Terra/Aqua surface-reflectance I/O.",
)

VIIRS_ADAPTER = SensorIOAdapter(
    sensor="viirs",
    supported_platforms=("snpp", "noaa20", "noaa21"),
    open_surface_reflectance=viirs.open_viirs_surface_reflectance,
    prepare_scene_for_inversion=viirs.prepare_viirs_scene_for_inversion,
    notes="VIIRS SNPP/NOAA-20/NOAA-21 surface-reflectance I/O.",
)

SENSOR_REGISTRY: dict[str, SensorIOAdapter] = {
    "modis": MODIS_ADAPTER,
    "viirs": VIIRS_ADAPTER,
}


def get_sensor_adapter(sensor: str, platform: str | None = None) -> SensorIOAdapter:
    """Return the adapter for a sensor/platform pair."""
    canonical_sensor = normalize_sensor_name(sensor)
    canonical_platform = normalize_platform_name(canonical_sensor, platform)
    adapter = SENSOR_REGISTRY[canonical_sensor]
    if canonical_platform is not None and canonical_platform not in adapter.supported_platforms:
        raise ValueError(
            f"Platform {canonical_platform!r} is not supported for sensor {canonical_sensor!r}. "
            f"Supported platforms: {adapter.supported_platforms}"
        )
    return adapter


def list_supported_sensors() -> tuple[str, ...]:
    """Return the canonical sensor families currently registered."""
    return tuple(SENSOR_REGISTRY)


def list_supported_sensor_platforms() -> dict[str, tuple[str, ...]]:
    """Return canonical platform names for each registered sensor family."""
    return {sensor: adapter.supported_platforms for sensor, adapter in SENSOR_REGISTRY.items()}


def describe_sensor(sensor: str, platform: str | None = None) -> dict[str, Any]:
    """Return a compact metadata summary for a registered adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return {
        "sensor": adapter.sensor,
        "supported_platforms": adapter.supported_platforms,
        "notes": adapter.notes,
    }
