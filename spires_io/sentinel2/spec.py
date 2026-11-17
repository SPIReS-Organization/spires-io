"""Sentinel-2 L2A from Element84 Earth Search: flat ``B??.tif`` COGs + ``SCL.tif`` + ``item.json``."""
import json
from pathlib import Path

import numpy as np
import xarray as xr

from spires_io.sensor_spec import (
    DEFAULT_QA_EXCLUDE,
    QA_FLAGS,
    SceneLayoutError,
    SensorSpec,
    flags_to_layers,
    valid_from_flags,
)
from spires_io.solar import per_pixel_sza_provider

# Scene Classification (SCL) classes by QA flag. SCL has no cloud-buffer class, so
# ``cloud_buffer`` is absent from QA_FLAGS_AVAILABLE. Classes 4-5 (vegetation, not
# vegetated) and 7 (unclassified) carry no flag and stay valid.
_SCL_CLASSES = {
    "fill": (0,),
    "saturated": (1,),
    "cloud_shadow": (2, 3),   # cast shadow, cloud shadow
    "cloud": (8, 9),          # cloud medium / high probability
    "cirrus": (10,),
    "water": (6,),
    "snow": (11,),            # snow / ice
}
QA_FLAGS_AVAILABLE = tuple(f for f in QA_FLAGS if f != "cloud_buffer")


def decode_qa_flags(layers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One bool plane per SCL-expressible flag; ``cloud_buffer`` is filled as all-False."""
    scl = layers["scl"]
    flags = {name: np.isin(scl, classes) for name, classes in _SCL_CLASSES.items()}
    return flags_to_layers(flags, scl.shape)


def decode_qa(layers: dict[str, np.ndarray], exclude: frozenset[str] = DEFAULT_QA_EXCLUDE) -> np.ndarray:
    return valid_from_flags(decode_qa_flags(layers), exclude)


def read_item(scene_dir: Path) -> dict:
    """The STAC item saved next to the COGs. Refuses items whose DNs still carry the BOA offset.

    Element84 COGs are offset-harmonized even though the item advertises
    ``raster:bands[0].offset = -0.1``; applying it again drives most visible
    pixels negative. ``earthsearch:boa_offset_applied`` is the flag that makes
    ``offset = 0.0`` below correct, so its absence is an error, not a default.
    """
    path = scene_dir / "item.json"
    if not path.is_file():
        raise SceneLayoutError(f"item.json (STAC item) missing in {scene_dir}")
    item = json.loads(path.read_text())
    if item.get("properties", {}).get("earthsearch:boa_offset_applied") is not True:
        raise ValueError(
            f"{scene_dir.name}: earthsearch:boa_offset_applied is not true; "
            "this spec assumes offset-harmonized Element84 COGs"
        )
    return item


def read_acquisition_time(scene_dir: Path) -> np.datetime64:
    stamp = read_item(scene_dir)["properties"]["datetime"]          # "2025-04-20T18:54:03.375000Z"
    return np.datetime64(stamp.rstrip("Z").split(".")[0], "s")


def read_platform(scene_dir: Path) -> str:
    return read_item(scene_dir)["properties"]["platform"]           # "sentinel-2a" / "-2b" / "-2c"


_BANDS_10M = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"}
_BANDS_20M = {"red_edge1": "B05", "red_edge2": "B06", "red_edge3": "B07", "nir_narrow": "B8A", "swir1": "B11", "swir2": "B12"}
_WAVELENGTH_NM = {"blue": 492.4, "green": 559.8, "red": 664.6, "red_edge1": 704.1, "red_edge2": 740.5,
                  "red_edge3": 782.8, "nir": 832.8, "nir_narrow": 864.7, "swir1": 1613.7, "swir2": 2202.4}

SPEC = SensorSpec(
    sensor="sentinel2",
    platforms=("sentinel2a", "sentinel2b", "sentinel2c"),
    band_files={n: f"{c}.tif" for n, c in {**_BANDS_10M, **_BANDS_20M}.items()},
    band_resolution_m={**{n: 10 for n in _BANDS_10M}, **{n: 20 for n in _BANDS_20M}},
    band_wavelength_nm=_WAVELENGTH_NM,
    scale=1e-4,
    offset=0.0,            # see read_item
    nodata=0,
    qa_files={"scl": "SCL.tif"},
    decode_qa=decode_qa,
    decode_qa_flags=decode_qa_flags,
    qa_flags_available=QA_FLAGS_AVAILABLE,
    qa_source="SCL",
    sza_provider=per_pixel_sza_provider(read_acquisition_time),
    acquisition_time=read_acquisition_time,
    default_target_resolution_m=20,
)


def open_sentinel2_scene(scene_dir, **kwargs) -> xr.Dataset:
    from spires_io.raster_scene import open_raster_scene

    ds = open_raster_scene(Path(scene_dir), SPEC, **kwargs)
    ds.attrs["platform"] = read_platform(Path(scene_dir))
    return ds
