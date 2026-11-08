"""Sensor registry and adapter definitions for SPIReS I/O."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import xarray as xr

import spires_io.hls as hls
import spires_io.landsat as landsat
import spires_io.modis as modis
import spires_io.sentinel2 as sentinel2
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
    "landsat": "landsat",
    "landsat8": "landsat",
    "landsat9": "landsat",
    "lc08": "landsat",
    "lc09": "landsat",
    "oli": "landsat",
    "hlsl30": "hls_l30",
    "hlss30": "hls_s30",
    "sentinel2": "sentinel2",
    "s2": "sentinel2",
    "msi": "sentinel2",
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
    "landsat": {"landsat8": "landsat8", "lc08": "landsat8", "landsat9": "landsat9", "lc09": "landsat9"},
    "hls_l30": {"landsat8": "landsat8", "lc08": "landsat8", "landsat9": "landsat9", "lc09": "landsat9"},
    "hls_s30": {"sentinel2a": "sentinel2a", "s2a": "sentinel2a", "sentinel2b": "sentinel2b", "s2b": "sentinel2b",
                "sentinel2c": "sentinel2c", "s2c": "sentinel2c"},
    "sentinel2": {"sentinel2a": "sentinel2a", "s2a": "sentinel2a", "sentinel2b": "sentinel2b", "s2b": "sentinel2b",
                  "sentinel2c": "sentinel2c", "s2c": "sentinel2c"},
}


def _normalize_token(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")
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

LANDSAT_ADAPTER = SensorIOAdapter(
    sensor="landsat",
    supported_platforms=landsat.SPEC.platforms,
    open_surface_reflectance=landsat.open_landsat_scene,
    prepare_scene_for_inversion=landsat.open_landsat_scene,
    notes="Landsat 8/9 Collection 2 Level-2 SR scene directories (local).",
)

HLS_L30_ADAPTER = SensorIOAdapter(
    sensor="hls_l30",
    supported_platforms=hls.L30_SPEC.platforms,
    open_surface_reflectance=hls.open_hls_l30_scene,
    prepare_scene_for_inversion=hls.open_hls_l30_scene,
    notes="HLS v2.0 L30 (Landsat-derived, 30 m) granule directories (local).",
)

HLS_S30_ADAPTER = SensorIOAdapter(
    sensor="hls_s30",
    supported_platforms=hls.S30_SPEC.platforms,
    open_surface_reflectance=hls.open_hls_s30_scene,
    prepare_scene_for_inversion=hls.open_hls_s30_scene,
    notes="HLS v2.0 S30 (Sentinel-2-derived, 30 m) granule directories (local).",
)

SENTINEL2_ADAPTER = SensorIOAdapter(
    sensor="sentinel2",
    supported_platforms=sentinel2.SPEC.platforms,
    open_surface_reflectance=sentinel2.open_sentinel2_scene,
    prepare_scene_for_inversion=sentinel2.open_sentinel2_scene,
    notes="Sentinel-2 L2A in the Element84 Earth Search COG layout (local); default 20 m grid.",
)

SENSOR_REGISTRY: dict[str, SensorIOAdapter] = {
    "modis": MODIS_ADAPTER,
    "viirs": VIIRS_ADAPTER,
    "landsat": LANDSAT_ADAPTER,
    "hls_l30": HLS_L30_ADAPTER,
    "hls_s30": HLS_S30_ADAPTER,
    "sentinel2": SENTINEL2_ADAPTER,
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
