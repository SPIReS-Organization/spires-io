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
from spires_io.ancillary import load_ancillary_layer, load_ancillary_layers
from spires_io.background import load_background_reflectance
from spires_io.configs import (
    MaskConfig,
    PostprocessConfig,
    SceneManifest,
    SceneManifestItem,
    SpiresRunConfig,
)
from spires_io.geometry import (
    add_illumination_geometry,
    derive_illumination_geometry,
)
from spires_io.loader import SpiresDataLoader
from spires_io.masks import load_external_mask
from spires_io.spires_data import SpiresData
from spires_io.writer import SpiresDataWriter

__all__ = [
    "__version__",
    "SceneManifest",
    "SceneManifestItem",
    "MaskConfig",
    "PostprocessConfig",
    "SpiresData",
    "SpiresDataLoader",
    "SpiresDataWriter",
    "SpiresRunConfig",
    "add_illumination_geometry",
    "describe_sensor",
    "derive_illumination_geometry",
    "list_supported_sensor_platforms",
    "list_supported_sensors",
    "load_ancillary_layer",
    "load_ancillary_layers",
    "load_background_reflectance",
    "load_external_mask",
    "normalize_platform_name",
    "normalize_sensor_name",
    "open_surface_reflectance",
    "prepare_scene_for_inversion",
]
