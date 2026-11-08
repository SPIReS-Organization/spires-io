# spires_io/landsat/spec.py
"""Landsat 8/9 Collection 2 Level-2 surface reflectance (``LC08_L2SP_*`` / ``LC09_L2SP_*``)."""
import re
from pathlib import Path

import numpy as np
import xarray as xr

from spires_io.sensor_spec import SceneLayoutError, SensorSpec
from spires_io.solar import per_pixel_sza_provider

# QA_PIXEL bits (Landsat Collection 2 Level-2 Science Product Guide, table 6-3).
_QA_PIXEL_INVALID_BITS = (0, 1, 2, 3, 4)   # fill, dilated cloud, cirrus, cloud, cloud shadow
# QA_RADSAT bit n-1 marks band n saturated; we use bands 2-7.
_QA_RADSAT_INVALID_BITS = (1, 2, 3, 4, 5, 6)


def _any_bit_set(values: np.ndarray, bits) -> np.ndarray:
    mask = sum(1 << b for b in bits)
    return (values & mask) != 0


def decode_qa(layers: dict[str, np.ndarray]) -> np.ndarray:
    """True where the pixel may be inverted (not fill/cloud/shadow/cirrus and not saturated)."""
    bad = _any_bit_set(layers["qa_pixel"], _QA_PIXEL_INVALID_BITS)
    saturated = _any_bit_set(layers["qa_radsat"], _QA_RADSAT_INVALID_BITS)
    return ~(bad | saturated)


def _mtl_path(scene_dir: Path) -> Path:
    matches = sorted(scene_dir.glob("*_MTL.txt"))
    if len(matches) != 1:
        raise SceneLayoutError(f"expected one *_MTL.txt in {scene_dir}, found {len(matches)}")
    return matches[0]


def _mtl_value(text: str, key: str) -> str:
    match = re.search(rf"^\s*{key}\s*=\s*\"?([^\"\n]+)\"?\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"{key} not found in MTL")
    return match.group(1).strip()


def read_acquisition_time(scene_dir: Path) -> np.datetime64:
    text = _mtl_path(scene_dir).read_text()
    date = _mtl_value(text, "DATE_ACQUIRED")
    clock = _mtl_value(text, "SCENE_CENTER_TIME").rstrip("Z").split(".")[0]
    return np.datetime64(f"{date}T{clock}", "s")


def read_platform(scene_dir: Path) -> str:
    return _mtl_value(_mtl_path(scene_dir).read_text(), "SPACECRAFT_ID")   # "LANDSAT_8" / "LANDSAT_9"


SPEC = SensorSpec(
    sensor="landsat",
    platforms=("landsat8", "landsat9"),
    band_files={
        "blue": "*_SR_B2.TIF", "green": "*_SR_B3.TIF", "red": "*_SR_B4.TIF",
        "nir": "*_SR_B5.TIF", "swir1": "*_SR_B6.TIF", "swir2": "*_SR_B7.TIF",
    },
    band_resolution_m={b: 30 for b in ("blue", "green", "red", "nir", "swir1", "swir2")},
    band_wavelength_nm={"blue": 482.0, "green": 561.0, "red": 655.0, "nir": 865.0, "swir1": 1609.0, "swir2": 2201.0},
    scale=2.75e-5,
    offset=-0.2,
    nodata=0,
    qa_files={"qa_pixel": "*_QA_PIXEL.TIF", "qa_radsat": "*_QA_RADSAT.TIF"},
    decode_qa=decode_qa,
    sza_provider=per_pixel_sza_provider(read_acquisition_time),
    acquisition_time=read_acquisition_time,
    default_target_resolution_m=30,
)


def open_landsat_scene(scene_dir, **kwargs) -> xr.Dataset:
    """Open a Landsat C2 L2 scene directory as an inversion-ready Dataset."""
    from spires_io.raster_scene import open_raster_scene

    ds = open_raster_scene(Path(scene_dir), SPEC, **kwargs)
    ds.attrs["platform"] = read_platform(Path(scene_dir))
    return ds
