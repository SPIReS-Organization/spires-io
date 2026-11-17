# spires_io/sensor_spec.py
"""Declarative per-sensor description of a raster scene, plus grid resamplers."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rioxarray  # noqa: F401  # register the .rio accessor
import xarray as xr
from spires_contract.qa import QA_FLAGS, QA_INVALID_FLAGS


class SceneLayoutError(FileNotFoundError):
    """A band or metadata file required by the spec is missing or ambiguous."""


class GridMismatchError(ValueError):
    """Bands do not share one grid after resampling."""


# All native-resolution bands keyed by common name, target resolution in metres
# -> all bands on one shared grid at that resolution.
Resampler = Callable[[dict[str, xr.DataArray], int], dict[str, xr.DataArray]]

# Sensor-neutral QA flag names, owned by spires-contract so the io and qa_mask
# layers cannot drift apart. Each spec's ``decode_qa`` maps the ones its QA
# product carries onto bits or classes; names the product lacks are no-ops
# (Sentinel-2 SCL has no cloud buffer). ``cloud_buffer`` is Landsat C2 "dilated
# cloud" and HLS Fmask "adjacent to cloud/shadow". ``snow`` and ``water`` are
# descriptive and never invalidate a pixel, so they are not in the default exclude.
DEFAULT_QA_EXCLUDE = frozenset(QA_INVALID_FLAGS)
QaDecoder = Callable[[dict[str, np.ndarray], frozenset[str]], np.ndarray]
QaFlagDecoder = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def normalize_qa_exclude(exclude) -> frozenset[str]:
    names = frozenset(exclude)
    unknown = names - set(QA_FLAGS)
    if unknown:
        raise ValueError(f"unknown qa_exclude flag(s) {sorted(unknown)}; choose from {QA_FLAGS}")
    return names


def flags_to_layers(flags: dict[str, np.ndarray], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Fill every QA_FLAGS entry: missing flags become all-False planes."""
    unknown = set(flags) - set(QA_FLAGS)
    if unknown:
        raise ValueError(f"decoder produced flag(s) not in QA_FLAGS: {sorted(unknown)}")
    return {f: flags.get(f, np.zeros(shape, dtype=bool)) for f in QA_FLAGS}


def valid_from_flags(flags: dict[str, np.ndarray], exclude: frozenset[str]) -> np.ndarray:
    """True where none of the ``exclude`` flags is set. Every sensor's ``decode_qa`` goes through this.

    Names outside QA_FLAGS are ignored, as the decoders did before the flag
    planes existed: ``exclude={"haze"}`` excludes nothing.
    """
    bad = np.zeros(flags["fill"].shape, dtype=bool)
    for name in exclude & set(QA_FLAGS):
        bad |= flags[name]
    return ~bad


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
    decode_qa: QaDecoder                    # ({layer: int array}, exclude flags) -> bool valid
    decode_qa_flags: QaFlagDecoder          # ({layer: int array}) -> {flag: bool array}, every QA_FLAGS entry
    qa_flags_available: tuple[str, ...]     # flags the product can express; the rest are all-False
    qa_source: str                          # provenance of the flags, e.g. "QA_PIXEL", "Fmask", "SCL"
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
