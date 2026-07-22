"""spires-io: data loaders and coordinate transforms for the SPIReS package family."""

from spires_contract import SpiresData

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
from spires_io.clustering import cluster
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
from spires_io.loader import SpiresDataLoader, load
from spires_io.masks import (
    assign_inversion_exclusion_masks,
    decode_inversion_exclusions,
    load_external_mask,
    pack_inversion_exclusions,
)
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
    "assign_inversion_exclusion_masks",
    "cluster",
    "decode_inversion_exclusions",
    "describe_sensor",
    "derive_illumination_geometry",
    "list_supported_sensor_platforms",
    "list_supported_sensors",
    "load",
    "load_ancillary_layer",
    "load_ancillary_layers",
    "load_background_reflectance",
    "load_external_mask",
    "normalize_platform_name",
    "normalize_sensor_name",
    "open_surface_reflectance",
    "pack_inversion_exclusions",
    "prepare_scene_for_inversion",
]
