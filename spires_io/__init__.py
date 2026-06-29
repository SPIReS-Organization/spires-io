"""spires-io: data loaders and coordinate transforms for the SPIReS package family."""

__version__ = "0.1.0"

from spires_io.api import (
    describe_sensor,
    list_supported_sensor_platforms,
    list_supported_sensors,
    normalize_platform_name,
    normalize_sensor_name,
    open_surface_reflectance,
    prepare_scene_for_inversion,
)
from spires_io.configs import SceneManifest, SceneManifestItem, SpiresRunConfig
from spires_io.spires_data import SpiresData

__all__ = [
    "__version__",
    "SceneManifest",
    "SceneManifestItem",
    "SpiresData",
    "SpiresRunConfig",
    "describe_sensor",
    "list_supported_sensor_platforms",
    "list_supported_sensors",
    "normalize_platform_name",
    "normalize_sensor_name",
    "open_surface_reflectance",
    "prepare_scene_for_inversion",
]
