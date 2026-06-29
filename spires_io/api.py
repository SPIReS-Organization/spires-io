"""Generic sensor-dispatch API for SPIReS I/O."""

from pathlib import Path

import xarray as xr

from spires_io.registry import (
    describe_sensor,
    get_sensor_adapter,
    list_supported_sensor_platforms,
    list_supported_sensors,
    normalize_platform_name,
    normalize_sensor_name,
)


def open_surface_reflectance(
    source: str | Path,
    *,
    sensor: str,
    platform: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Open a surface-reflectance source using the registered sensor adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.open_surface_reflectance(source, **kwargs)


def prepare_scene_for_inversion(
    source,
    *,
    sensor: str,
    platform: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Prepare a scene for inversion using the registered sensor adapter."""
    adapter = get_sensor_adapter(sensor, platform)
    return adapter.prepare_scene_for_inversion(source, **kwargs)


__all__ = [
    "describe_sensor",
    "list_supported_sensor_platforms",
    "list_supported_sensors",
    "normalize_platform_name",
    "normalize_sensor_name",
    "open_surface_reflectance",
    "prepare_scene_for_inversion",
]
