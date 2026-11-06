"""HLS v2.0 (``HLS.L30.*`` / ``HLS.S30.*``) surface reflectance, 30 m, UTM MGRS tiles."""
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from spires_io.sensor_spec import SensorSpec, nearest, resolution_m

# Fmask bits (HLS v2.0 User Guide, table 9).
_FMASK_INVALID_BITS = (0, 1, 2, 3)   # cirrus, cloud, adjacent to cloud/shadow, cloud shadow
_FMASK_FILL = 255


def decode_qa(layers: dict[str, np.ndarray]) -> np.ndarray:
    fmask = layers["fmask"]
    mask = sum(1 << b for b in _FMASK_INVALID_BITS)
    return ((fmask & mask) == 0) & (fmask != _FMASK_FILL)


def _first_band_path(scene_dir: Path) -> Path:
    matches = sorted(scene_dir.glob("*.B02.tif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one *.B02.tif in {scene_dir}, found {len(matches)}")
    return matches[0]


def read_acquisition_time(scene_dir: Path) -> np.datetime64:
    """``SENSING_TIME`` GeoTIFF tag; S30 may list two granule times separated by ';' — take the first."""
    with rasterio.open(_first_band_path(scene_dir)) as src:
        sensing = src.tags().get("SENSING_TIME")
    if not sensing:
        raise ValueError(f"SENSING_TIME tag missing in {scene_dir}")
    first = sensing.split(";")[0].strip().rstrip("Z").split(".")[0]
    return np.datetime64(first, "s")


def shipped_sza(scene_dir: Path, template: xr.DataArray) -> xr.DataArray:
    """HLS ships a per-pixel SZA band in hundredths of a degree (uint16)."""
    from spires_io.raster_scene import read_band, resolve_one

    sza = read_band(resolve_one(scene_dir, "sza", "*.SZA.tif"))
    sza = nearest({"sza": sza}, resolution_m(template))["sza"]
    return sza.astype(np.float32) * np.float32(0.01)


_L30_BANDS = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "nir": "B05", "swir1": "B06", "swir2": "B07"}
_S30_BANDS = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "red_edge1": "B05",
              "red_edge2": "B06", "red_edge3": "B07", "nir": "B08", "nir_narrow": "B8A",
              "swir1": "B11", "swir2": "B12"}
_L30_WAVELENGTH_NM = {"coastal": 443.0, "blue": 482.0, "green": 561.0, "red": 655.0, "nir": 865.0, "swir1": 1609.0, "swir2": 2201.0}
_S30_WAVELENGTH_NM = {"coastal": 442.7, "blue": 492.4, "green": 559.8, "red": 664.6, "red_edge1": 704.1,
                      "red_edge2": 740.5, "red_edge3": 782.8, "nir": 832.8, "nir_narrow": 864.7,
                      "swir1": 1613.7, "swir2": 2202.4}


def _spec(sensor, platforms, bands, wavelengths) -> SensorSpec:
    return SensorSpec(
        sensor=sensor,
        platforms=platforms,
        band_files={name: f"*.{code}.tif" for name, code in bands.items()},
        band_resolution_m={name: 30 for name in bands},
        band_wavelength_nm=wavelengths,
        scale=1e-4,
        offset=0.0,
        nodata=-9999,
        qa_files={"fmask": "*.Fmask.tif"},
        decode_qa=decode_qa,
        sza_provider=shipped_sza,
        acquisition_time=read_acquisition_time,
        default_target_resolution_m=30,
    )


L30_SPEC = _spec("hls_l30", ("landsat8", "landsat9"), _L30_BANDS, _L30_WAVELENGTH_NM)
S30_SPEC = _spec("hls_s30", ("sentinel2a", "sentinel2b", "sentinel2c"), _S30_BANDS, _S30_WAVELENGTH_NM)


def open_hls_l30_scene(scene_dir, **kwargs) -> xr.Dataset:
    from spires_io.raster_scene import open_raster_scene
    return open_raster_scene(Path(scene_dir), L30_SPEC, **kwargs)


def open_hls_s30_scene(scene_dir, **kwargs) -> xr.Dataset:
    from spires_io.raster_scene import open_raster_scene
    return open_raster_scene(Path(scene_dir), S30_SPEC, **kwargs)
