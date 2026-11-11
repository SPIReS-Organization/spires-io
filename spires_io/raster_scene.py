# spires_io/raster_scene.py
"""Open a local raster scene directory as an inversion-ready Dataset, driven by a SensorSpec."""
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from spires_contract.spectra import validate_solar_angles, validate_target_spectra

from spires_io.sensor_spec import (
    GridMismatchError,
    Resampler,
    SceneLayoutError,
    SensorSpec,
    block_mean,
    nearest,
    resolution_m,
)


def resolve_one(scene_dir: Path, name: str, pattern: str) -> Path:
    matches = sorted(scene_dir.glob(pattern))
    if len(matches) != 1:
        raise SceneLayoutError(
            f"band/layer {name!r}: pattern {pattern!r} has {len(matches)} matches in {scene_dir}"
        )
    return matches[0]


def read_band(path: Path) -> xr.DataArray:
    """Single-band GeoTIFF/JP2 as a 2-D (y, x) DataArray, raw DN dtype, no masking."""
    da = rioxarray.open_rasterio(path, masked=False)
    return da.squeeze("band", drop=True).load()


def clip_box(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """Clip a (y, x) raster to ``bbox = (minx, miny, maxx, maxy)`` in the raster's CRS."""
    from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster

    minx, miny, maxx, maxy = bbox
    if not (minx < maxx and miny < maxy):
        raise ValueError(f"bbox must be (minx, miny, maxx, maxy) with min < max, got {bbox}")
    try:
        return da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    except (NoDataInBounds, OneDimensionalRaster) as exc:
        raise ValueError(f"bbox {bbox} does not intersect the scene ({exc})") from exc


def to_reflectance(da: xr.DataArray, scale: float, offset: float, nodata) -> xr.DataArray:
    values = da.values.astype(np.float32) * np.float32(scale) + np.float32(offset)
    values[da.values == nodata] = np.nan
    return da.copy(data=values)


def check_same_grid(arrays: dict[str, xr.DataArray]) -> None:
    first_name, first = next(iter(arrays.items()))
    for name, da in arrays.items():
        if da.shape != first.shape or da.rio.transform() != first.rio.transform():
            raise GridMismatchError(
                f"{name!r} grid {da.shape} / {da.rio.transform()} differs from "
                f"{first_name!r} {first.shape} / {first.rio.transform()}"
            )


def open_raster_scene(
    scene_dir: Path,
    spec: SensorSpec,
    *,
    bands: list[str] | None = None,
    target_resolution_m: int | None = None,
    resampler: Resampler | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> xr.Dataset:
    """Read, scale, resample, QA-decode and validate one scene.

    Returns a Dataset with ``reflectance (y, x, band)`` float32 (NaN = nodata),
    ``valid (y, x)`` bool (True = run the inversion) and ``solar_zenith (y, x)``
    float32 degrees, all on one grid at ``target_resolution_m``. ``bbox`` is
    ``(minx, miny, maxx, maxy)`` in the scene CRS; bands and QA layers are
    clipped before resampling, so the output grid stays pixel-aligned with
    the full scene.
    """
    scene_dir = Path(scene_dir)
    names = list(bands) if bands is not None else list(spec.band_files)
    unknown = [n for n in names if n not in spec.band_files]
    if unknown:
        raise ValueError(f"unknown band(s) {unknown} for {spec.sensor}; spec defines {list(spec.band_files)}")
    target_m = target_resolution_m or spec.default_target_resolution_m

    # Fail on layout before any heavy read.
    band_paths = {n: resolve_one(scene_dir, n, spec.band_files[n]) for n in names}
    qa_paths = {n: resolve_one(scene_dir, n, spec.qa_files[n]) for n in spec.qa_files}
    acquisition_time = spec.acquisition_time(scene_dir)

    raw = {n: read_band(p) for n, p in band_paths.items()}
    if bbox is not None:
        raw = {n: clip_box(da, bbox) for n, da in raw.items()}
    scaled = {n: to_reflectance(da, spec.scale, spec.offset, spec.nodata) for n, da in raw.items()}

    if resampler is None:
        finest = min(resolution_m(da) for da in scaled.values())
        resampler = block_mean if target_m >= finest else nearest
    on_grid = resampler(scaled, target_m)
    check_same_grid(on_grid)
    template = on_grid[names[0]]

    qa_raw = {n: read_band(p) for n, p in qa_paths.items()}
    if bbox is not None:
        qa_raw = {n: clip_box(da, bbox) for n, da in qa_raw.items()}
    qa_on_grid = nearest(qa_raw, target_m)
    check_same_grid({**qa_on_grid, "_reflectance": template})
    valid = spec.decode_qa({n: da.values for n, da in qa_on_grid.items()})
    stack = np.stack([on_grid[n].values for n in names], axis=-1)
    valid &= np.isfinite(stack).all(axis=-1)

    reflectance = xr.DataArray(
        stack.astype(np.float32),
        dims=("y", "x", "band"),
        coords={
            "y": template.y, "x": template.x, "band": names,
            "wavelength_nm": ("band", np.array([spec.band_wavelength_nm[n] for n in names], np.float32)),
        },
    )
    solar_zenith = spec.sza_provider(scene_dir, template).astype(np.float32)
    if solar_zenith.shape != template.shape:
        solar_zenith = solar_zenith.reindex_like(template, method="nearest", tolerance=target_m / 2)
    solar_zenith.attrs = {"units": "degree"}
    solar_zenith.encoding = {}

    validate_target_spectra(reflectance)
    validate_solar_angles(solar_zenith)

    ds = xr.Dataset(
        {
            "reflectance": reflectance,
            "valid": xr.DataArray(valid, dims=("y", "x"), coords={"y": template.y, "x": template.x}),
            "solar_zenith": solar_zenith,
        },
        attrs={
            "sensor": spec.sensor,
            "acquisition_time": str(acquisition_time),
            "resolution_m": target_m,
            "source_product_id": scene_dir.name,
        },
    )
    return ds.rio.write_crs(template.rio.crs)
