# spires_io/sensor_spec.py
"""Declarative per-sensor description of a raster scene, plus grid resamplers."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rioxarray  # noqa: F401  # register the .rio accessor
import xarray as xr


class SceneLayoutError(FileNotFoundError):
    """A band or metadata file required by the spec is missing or ambiguous."""


class GridMismatchError(ValueError):
    """Bands do not share one grid after resampling."""


# All native-resolution bands keyed by common name, target resolution in metres
# -> all bands on one shared grid at that resolution.
Resampler = Callable[[dict[str, xr.DataArray], int], dict[str, xr.DataArray]]


@dataclass(frozen=True)
class SensorSpec:
    sensor: str
    platforms: tuple[str, ...]
    band_files: dict[str, str]              # common name -> glob relative to scene_dir
    band_resolution_m: dict[str, int]
    band_wavelength_nm: dict[str, float]
    scale: float
    offset: float
    nodata: int | float
    qa_files: dict[str, str]                # QA layer name -> glob relative to scene_dir
    decode_qa: Callable[[dict[str, np.ndarray]], np.ndarray]           # {layer: int array} -> bool valid
    sza_provider: Callable[[Path, xr.DataArray], xr.DataArray]         # (scene_dir, grid template) -> degrees
    acquisition_time: Callable[[Path], np.datetime64]
    default_target_resolution_m: int


def resolution_m(da: xr.DataArray) -> int:
    """Pixel size in metres of a north-up DataArray with square pixels."""
    dx, dy = da.rio.resolution()
    if abs(abs(dx) - abs(dy)) > 1e-6:
        raise ValueError(f"non-square pixels: {dx} x {dy}")
    return int(round(abs(dx)))


def integer_factor(coarse_m: int, fine_m: int) -> int:
    """How many fine pixels span one coarse pixel; raises unless it is a whole number."""
    if coarse_m % fine_m != 0:
        raise ValueError(f"{coarse_m} m is not an integer multiple of {fine_m} m")
    return coarse_m // fine_m


def on_grid(values: np.ndarray, template: xr.DataArray, target_m: int) -> xr.DataArray:
    """Wrap ``values`` on the grid that shares ``template``'s upper-left corner at ``target_m``."""
    transform = template.rio.transform()
    ny, nx = values.shape
    x = transform.c + target_m * (np.arange(nx) + 0.5)
    y = transform.f - target_m * (np.arange(ny) + 0.5)
    out = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, attrs=template.attrs)
    return out.rio.write_crs(template.rio.crs)


def block_mean(bands: dict[str, xr.DataArray], target_m: int) -> dict[str, xr.DataArray]:
    """Downsample by averaging f x f blocks, ignoring NaN. Bands already at target pass through."""
    out = {}
    for name, da in bands.items():
        native = resolution_m(da)
        if native == target_m:
            out[name] = da
            continue
        if native > target_m:
            raise ValueError(f"block_mean needs a coarser target: band {name!r} is {native} m, target {target_m} m")
        f = integer_factor(target_m, native)
        ny, nx = da.shape
        if ny % f or nx % f:
            raise ValueError(f"band {name!r} shape {da.shape} is not divisible by factor {f}")
        blocks = da.values.reshape(ny // f, f, nx // f, f)
        finite = np.isfinite(blocks)
        count = finite.sum(axis=(1, 3))
        total = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
        mean = np.full(count.shape, np.nan, dtype=np.float32)
        np.divide(total, count, out=mean, where=count > 0)
        out[name] = on_grid(mean, da, target_m)
    return out


def nearest(bands: dict[str, xr.DataArray], target_m: int) -> dict[str, xr.DataArray]:
    """Nearest-neighbour to ``target_m``: repeat pixels when upsampling, take block centres when downsampling."""
    out = {}
    for name, da in bands.items():
        native = resolution_m(da)
        if native == target_m:
            out[name] = da
            continue
        if native > target_m:
            f = integer_factor(native, target_m)
            values = da.values.repeat(f, axis=0).repeat(f, axis=1)
        else:
            f = integer_factor(target_m, native)
            values = da.values[f // 2::f, f // 2::f]
        out[name] = on_grid(values, da, target_m)
    return out
