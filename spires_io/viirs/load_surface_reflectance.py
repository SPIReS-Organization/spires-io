"""Reader and scene-prep helpers for VIIRS VNP09GA / VJ109GA / VJ209GA products."""

from dataclasses import asdict
from datetime import datetime, timedelta
import logging
import re
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import xarray as xr

from spires_io.logging_utils import log_event
from spires_io.base import SceneMetadata, collect_attrs, normalize_path, read_scaled_array
from spires_io.masks import load_external_mask_on_grid
from spires_io.viirs.bands import (
    VIIRS_1KM_REFLECTANCE_BANDS,
    VIIRS_500M_REFLECTANCE_BANDS,
    partition_viirs_band_names,
    reflectance_field_name,
    resolve_viirs_inversion_bands_with_source,
)
from spires_io.viirs.fields import (
    VIIRS_1KM_GEOMETRY_FIELDS,
    VIIRS_1KM_GRID,
    VIIRS_1KM_GRID_NAME,
    VIIRS_1KM_QA_FIELDS,
    VIIRS_500M_GRID,
    VIIRS_500M_GRID_NAME,
    VIIRS_500M_SUPPORT_FIELDS,
)
from spires_io.viirs.geospatial import attach_spatial_ref, copy_spatial_metadata, parse_viirs_grid_metadata
from spires_io.viirs.qa import decode_viirs_qa_masks, load_external_cloud_masks


VIIRS_FILENAME_RE = re.compile(
    r"^(?P<product>VNP09GA|VJ109GA|VJ209GA)\.A(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?P<tile>h\d{2}v\d{2})\.(?P<collection>\d{3})\.(?P<processing>\d+)\.h5$"
)

PLATFORM_BY_PRODUCT = {
    "VNP09GA": "snpp",
    "VJ109GA": "noaa20",
    "VJ209GA": "noaa21",
}

LOGGER = logging.getLogger(__name__)


def parse_viirs_surface_reflectance_filename(path: str | Path) -> SceneMetadata:
    """Parse standard VNP09GA / VJ109GA / VJ209GA filenames into normalized metadata."""
    path = normalize_path(path)
    match = VIIRS_FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized VIIRS surface reflectance filename: {path.name}")

    product = match.group("product")
    year = int(match.group("year"))
    doy = int(match.group("doy"))
    acquisition_date = (datetime(year, 1, 1) + timedelta(days=doy - 1)).date().isoformat()

    return SceneMetadata(
        product=product,
        platform=PLATFORM_BY_PRODUCT[product],
        tile=match.group("tile"),
        acquisition_date=acquisition_date,
        collection=match.group("collection"),
        processing_timestamp=match.group("processing"),
        source_path=str(path),
    )


def _data_field_path(grid_path: str, dataset_name: str) -> str:
    return f"{grid_path}/Data Fields/{dataset_name}"


def _coord_path(grid_path: str, coord_name: str) -> str:
    return f"{grid_path}/{coord_name}"


def _open_band_stack(
    hdf: h5py.File,
    grid_path: str,
    band_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, str]]:
    arrays = []
    units_by_band = {}
    for band_name in band_names:
        dataset = hdf[_data_field_path(grid_path, reflectance_field_name(band_name))]
        arrays.append(read_scaled_array(dataset))
        units_by_band[band_name] = collect_attrs(dataset).get("units", "")
    stacked = np.stack(arrays, axis=-1)
    return stacked, units_by_band


def _empty_band_stack(y_coords: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
    """Return an empty reflectance cube with the native spatial shape."""
    return np.empty((y_coords.size, x_coords.size, 0), dtype=np.float32)


def _open_scalar_fields(
    hdf: h5py.File,
    grid_path: str,
    field_map: dict[str, str],
    *,
    apply_scale: bool,
    mask_fill: bool,
    mask_valid_range: bool,
) -> dict[str, xr.DataArray]:
    result = {}
    for variable_name, dataset_name in field_map.items():
        dataset = hdf[_data_field_path(grid_path, dataset_name)]
        array = read_scaled_array(
            dataset,
            apply_scale=apply_scale,
            mask_fill=mask_fill,
            mask_valid_range=mask_valid_range,
        )
        attrs = collect_attrs(dataset)
        result[variable_name] = xr.DataArray(
            array,
            dims=("y_1km", "x_1km") if "1km" in grid_path else ("y_500m", "x_500m"),
            attrs=attrs,
        )
    return result


def open_viirs_surface_reflectance(
    path: str | Path,
    *,
    bands: tuple[str, ...] | list[str] | None = None,
    lut_file: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> xr.Dataset:
    """
    Open a single VIIRS VNP09GA, VJ109GA, or VJ209GA file as a normalized xarray dataset.

    The first version intentionally preserves the native split-grid structure:
    - 1 km moderate-band reflectance, geometry, and QA layers
    - 500 m imagery-band reflectance and support layers

    Parameters
    ----------
    path
        VIIRS `VNP09GA`, `VJ109GA`, or `VJ209GA` HDF path.
    bands
        Optional VIIRS band subset to read. If omitted, all available VIIRS
        reflectance bands are read unless ``lut_file`` is provided.
    lut_file
        Optional LUT path used to resolve the VIIRS band subset to read.
    logger
        Optional logger for structured workflow messages.
    """
    start_time = perf_counter()
    logger = logger or LOGGER
    path = normalize_path(path)
    scene = parse_viirs_surface_reflectance_filename(path)
    selected_bands, band_selection_source = resolve_viirs_inversion_bands_with_source(
        bands=bands,
        lut_file=lut_file,
    )
    selected_500m_bands, selected_1km_bands = partition_viirs_band_names(selected_bands)

    with h5py.File(path, "r") as hdf:
        grid_metadata_1km = parse_viirs_grid_metadata(hdf, VIIRS_1KM_GRID_NAME)
        grid_metadata_500m = parse_viirs_grid_metadata(hdf, VIIRS_500M_GRID_NAME)
        x_1km = np.array(hdf[_coord_path(VIIRS_1KM_GRID, "XDim")][...])
        y_1km = np.array(hdf[_coord_path(VIIRS_1KM_GRID, "YDim")][...])
        x_500m = np.array(hdf[_coord_path(VIIRS_500M_GRID, "XDim")][...])
        y_500m = np.array(hdf[_coord_path(VIIRS_500M_GRID, "YDim")][...])

        if selected_1km_bands:
            reflectance_1km, units_1km = _open_band_stack(hdf, VIIRS_1KM_GRID, tuple(selected_1km_bands))
        else:
            reflectance_1km = _empty_band_stack(y_1km, x_1km)
            units_1km = {}

        if selected_500m_bands:
            reflectance_500m, units_500m = _open_band_stack(hdf, VIIRS_500M_GRID, tuple(selected_500m_bands))
        else:
            reflectance_500m = _empty_band_stack(y_500m, x_500m)
            units_500m = {}

        ds = xr.Dataset(
            data_vars={
                "reflectance_1km": xr.DataArray(
                    reflectance_1km,
                    dims=("y_1km", "x_1km", "band_1km"),
                    coords={
                        "y_1km": y_1km,
                        "x_1km": x_1km,
                        "band_1km": selected_1km_bands,
                    },
                    attrs={"units_by_band": units_1km},
                ),
                "reflectance_500m": xr.DataArray(
                    reflectance_500m,
                    dims=("y_500m", "x_500m", "band_500m"),
                    coords={
                        "y_500m": y_500m,
                        "x_500m": x_500m,
                        "band_500m": selected_500m_bands,
                    },
                    attrs={"units_by_band": units_500m},
                ),
            },
            coords={
                "x_1km": x_1km,
                "y_1km": y_1km,
                "x_500m": x_500m,
                "y_500m": y_500m,
                "band_1km": selected_1km_bands,
                "band_500m": selected_500m_bands,
            },
            attrs=asdict(scene),
        )

        ds.update(
            _open_scalar_fields(
                hdf,
                VIIRS_1KM_GRID,
                VIIRS_1KM_GEOMETRY_FIELDS,
                apply_scale=True,
                mask_fill=True,
                mask_valid_range=True,
            )
        )
        ds.update(
            _open_scalar_fields(
                hdf,
                VIIRS_1KM_GRID,
                VIIRS_1KM_QA_FIELDS,
                apply_scale=False,
                mask_fill=False,
                mask_valid_range=False,
            )
        )
        ds.update(
            _open_scalar_fields(
                hdf,
                VIIRS_500M_GRID,
                VIIRS_500M_SUPPORT_FIELDS,
                apply_scale=False,
                mask_fill=False,
                mask_valid_range=False,
            )
        )

        ds["reflectance_1km"].attrs["long_name"] = "VIIRS 1 km surface reflectance"
        ds["reflectance_500m"].attrs["long_name"] = "VIIRS 500 m surface reflectance"
        ds["x_1km"].attrs.update(collect_attrs(hdf[_coord_path(VIIRS_1KM_GRID, "XDim")]))
        ds["y_1km"].attrs.update(collect_attrs(hdf[_coord_path(VIIRS_1KM_GRID, "YDim")]))
        ds["x_500m"].attrs.update(collect_attrs(hdf[_coord_path(VIIRS_500M_GRID, "XDim")]))
        ds["y_500m"].attrs.update(collect_attrs(hdf[_coord_path(VIIRS_500M_GRID, "YDim")]))
        if grid_metadata_1km is not None:
            ds["reflectance_1km"].attrs.update(grid_metadata_1km.to_attrs())
        if grid_metadata_500m is not None:
            ds["reflectance_500m"].attrs.update(grid_metadata_500m.to_attrs())
        ds.attrs["selected_bands"] = selected_bands
        ds.attrs["band_selection_source"] = band_selection_source
        if lut_file is not None:
            ds.attrs["lut_file"] = str(lut_file)

        ds = attach_spatial_ref(
            ds,
            x_dim="x_500m",
            y_dim="y_500m",
            grid_metadata=grid_metadata_500m,
            data_var_names=("reflectance_500m",),
        )

    log_event(
        logger,
        "open_viirs_surface_reflectance",
        input_path=str(path),
        product=scene.product,
        platform=scene.platform,
        tile=scene.tile,
        acquisition_date=scene.acquisition_date,
        lut_file=str(lut_file) if lut_file is not None else None,
        selected_bands=selected_bands,
        selected_500m_bands=selected_500m_bands,
        selected_1km_bands=selected_1km_bands,
        band_selection_source=band_selection_source,
        elapsed_seconds=round(perf_counter() - start_time, 6),
    )

    return ds


def _upsample_1km_array_to_500m(array: np.ndarray) -> np.ndarray:
    """Upsample a native 1 km VIIRS array to the 500 m grid via nearest-neighbor."""
    return np.repeat(np.repeat(array, 2, axis=0), 2, axis=1)


def _upsample_1km_dataarray_to_500m(data_array: xr.DataArray, *, target_x: xr.DataArray, target_y: xr.DataArray) -> xr.DataArray:
    """Expand a 1 km DataArray to the 500 m grid using exact 2x nearest-neighbor replication."""
    array = np.asarray(data_array.values)
    upsampled = _upsample_1km_array_to_500m(array)

    dims = []
    coords = {}
    for dim in data_array.dims:
        if dim == "y_1km":
            dims.append("y")
            coords["y"] = target_y.values
        elif dim == "x_1km":
            dims.append("x")
            coords["x"] = target_x.values
        else:
            dims.append(dim)
            coords[dim] = data_array.coords[dim].values

    return xr.DataArray(upsampled, dims=dims, coords=coords, attrs=data_array.attrs.copy())


def _normalize_500m_dataarray(data_array: xr.DataArray) -> xr.DataArray:
    """Rename native 500 m dimensions to the common prepared-scene grid names."""
    rename_map = {}
    if "y_500m" in data_array.dims:
        rename_map["y_500m"] = "y"
    if "x_500m" in data_array.dims:
        rename_map["x_500m"] = "x"
    if "band_500m" in data_array.dims:
        rename_map["band_500m"] = "band"
    return data_array.rename(rename_map)


def _build_component_masks(
    reflectance: xr.DataArray,
    sensor_zenith: xr.DataArray,
    solar_zenith: xr.DataArray,
    land_water_mask: xr.DataArray,
    num_observations_1km: xr.DataArray,
    num_observations_500m: xr.DataArray,
    mask_cloud: xr.DataArray,
    mask_cloud_shadow: xr.DataArray,
    mask_snow: xr.DataArray,
    mask_water_external: xr.DataArray | None = None,
    mask_ice_external: xr.DataArray | None = None,
    mask_playa_external: xr.DataArray | None = None,
    *,
    water_mask_values: tuple[int, ...],
    mask_water_using_reflectance_qf: bool,
    mask_low_reflectance_for_inversion: bool,
    low_reflectance_threshold: float,
    max_sensor_zenith: float,
    max_solar_zenith: float,
    min_obs_1km: int,
    min_obs_500m: int,
) -> xr.Dataset:
    """Build transparent component masks and final valid masks on the 500 m grid."""
    finite_reflectance = np.isfinite(reflectance)
    mask_invalid_reflectance = ~finite_reflectance.all(dim="band")

    mask_bad_geometry = (
        (~np.isfinite(sensor_zenith))
        | (~np.isfinite(solar_zenith))
        | (sensor_zenith > max_sensor_zenith)
        | (solar_zenith > max_solar_zenith)
    )

    mask_shape = land_water_mask.shape
    mask_dims = land_water_mask.dims
    mask_coords = {dim: land_water_mask.coords[dim].values for dim in mask_dims}
    false_mask = xr.DataArray(np.zeros(mask_shape, dtype=bool), dims=mask_dims, coords=mask_coords)

    mask_water_reflectance_qf = false_mask.copy()
    if mask_water_using_reflectance_qf:
        for value in water_mask_values:
            mask_water_reflectance_qf = mask_water_reflectance_qf | (land_water_mask == value)
    mask_water_external = (
        false_mask.copy()
        if mask_water_external is None
        else mask_water_external.astype(bool)
    )
    mask_ice_external = (
        false_mask.copy()
        if mask_ice_external is None
        else mask_ice_external.astype(bool)
    )
    mask_playa_external = (
        false_mask.copy()
        if mask_playa_external is None
        else mask_playa_external.astype(bool)
    )
    mask_water = mask_water_reflectance_qf | mask_water_external

    if mask_low_reflectance_for_inversion:
        mask_low_reflectance = (reflectance < low_reflectance_threshold).all(dim="band")
    else:
        mask_low_reflectance = false_mask.copy()

    mask_low_observation_support = (
        (num_observations_1km < min_obs_1km)
        | (num_observations_500m < min_obs_500m)
    )

    valid_inversion_mask = ~(
        mask_invalid_reflectance
        | mask_bad_geometry
        | mask_water
        | mask_low_observation_support
        | mask_cloud
        | mask_cloud_shadow
        | mask_ice_external
        | mask_playa_external
        | mask_low_reflectance
    )
    valid_r0_mask = ~(
        mask_invalid_reflectance
        | mask_bad_geometry
        | mask_low_observation_support
        | mask_cloud
        | mask_cloud_shadow
        | mask_ice_external
        | mask_playa_external
        | mask_snow
    )

    return xr.Dataset(
        data_vars={
            "mask_invalid_reflectance": mask_invalid_reflectance.astype(bool),
            "mask_bad_geometry": mask_bad_geometry.astype(bool),
            "mask_water_reflectance_qf": mask_water_reflectance_qf.astype(bool),
            "mask_water_external": mask_water_external.astype(bool),
            "mask_water": mask_water.astype(bool),
            "mask_low_observation_support": mask_low_observation_support.astype(bool),
            "mask_cloud": mask_cloud.astype(bool),
            "mask_cloud_shadow": mask_cloud_shadow.astype(bool),
            "mask_snow": mask_snow.astype(bool),
            "mask_ice_external": mask_ice_external.astype(bool),
            "mask_playa_external": mask_playa_external.astype(bool),
            "mask_low_reflectance": mask_low_reflectance.astype(bool),
            "valid_inversion_mask": valid_inversion_mask.astype(bool),
            "valid_r0_mask": valid_r0_mask.astype(bool),
        }
    )


def prepare_viirs_scene_for_inversion(
    source: str | Path | xr.Dataset,
    *,
    bands: tuple[str, ...] | list[str] | None = None,
    lut_file: str | Path | None = None,
    logger: logging.Logger | None = None,
    cloud_mask_source: str | Path | xr.Dataset | xr.DataArray | None = None,
    cloud_mask_var: str = "mask_cloud",
    cloud_shadow_mask_var: str = "mask_cloud_shadow",
    water_mask_source: str | Path | xr.Dataset | xr.DataArray | None = None,
    water_mask_var: str | None = None,
    ice_mask_source: str | Path | xr.Dataset | xr.DataArray | None = None,
    ice_mask_var: str | None = None,
    playa_mask_source: str | Path | xr.Dataset | xr.DataArray | None = None,
    playa_mask_var: str | None = None,
    keep_intermediate_reflectance: bool = False,
    max_sensor_zenith: float = 65.0,
    max_solar_zenith: float = 85.0,
    min_obs_1km: int = 1,
    min_obs_500m: int = 1,
    water_mask_values: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 7),
    mask_water_using_reflectance_qf: bool = True,
    mask_water_using_external_file: bool = True,
    mask_low_reflectance_for_inversion: bool = False,
    low_reflectance_threshold: float = 0.1,
    write_detailed_masks: bool = False,
) -> xr.Dataset:
    """
    Prepare a VIIRS scene on a single 500 m analysis grid for downstream inversion.

    Parameters
    ----------
    source
        Either a VIIRS file path or the output from ``open_viirs_surface_reflectance``.
    bands
        Output band order for the merged 500 m analysis cube. If omitted, the
        full reflective VIIRS set is used unless ``lut_file`` is provided.
    lut_file
        Optional LUT path used to infer the VIIRS band subset from the LUT
        filename. Explicit ``bands`` takes precedence over this.
    logger
        Optional logger for structured workflow messages.
    cloud_mask_source
        Optional external cloud-mask source. If provided, this is combined
        with the QA-decoded cloud and cloud-shadow masks. Accepted inputs are
        a path to an xarray-readable file, an ``xr.Dataset``, or an
        ``xr.DataArray``.
    cloud_mask_var
        Variable name to read as the cloud mask when ``cloud_mask_source`` is a
        dataset-like object.
    cloud_shadow_mask_var
        Variable name to read as the cloud-shadow mask when
        ``cloud_mask_source`` is a dataset-like object.
    keep_intermediate_reflectance
        If True, retain intermediate reflectance cubes for debugging:
        ``reflectance_500m_native`` and ``reflectance_1km_on_500m``.
    max_sensor_zenith
        Pixels above this view angle are masked in ``valid_inversion_mask``.
    max_solar_zenith
        Pixels above this solar zenith are masked in ``valid_inversion_mask``.
    min_obs_1km
        Minimum 1 km observation support threshold.
    min_obs_500m
        Minimum 500 m observation support threshold.
    water_mask_values
        Values in ``land_water_mask`` that should be excluded as water or
        mixed water for inversion masking. By default this includes all
        VIIRS classes except land.
    """
    start_time = perf_counter()
    logger = logger or LOGGER
    if low_reflectance_threshold < 0:
        raise ValueError("low_reflectance_threshold must be >= 0")

    if isinstance(source, xr.Dataset):
        raw = source
    else:
        raw = open_viirs_surface_reflectance(source, bands=bands, lut_file=lut_file, logger=logger)

    bands, band_selection_source = resolve_viirs_inversion_bands_with_source(bands=bands, lut_file=lut_file)

    x = raw["x_500m"]
    y = raw["y_500m"]

    moderate_500m = _upsample_1km_dataarray_to_500m(raw["reflectance_1km"], target_x=x, target_y=y)
    moderate_500m = moderate_500m.rename({"band_1km": "band"})
    imagery_500m = _normalize_500m_dataarray(raw["reflectance_500m"])

    merged_reflectance = xr.concat([imagery_500m, moderate_500m], dim="band")
    merged_reflectance = merged_reflectance.sel(band=bands).transpose("y", "x", "band")
    merged_reflectance.name = "reflectance"
    merged_reflectance.attrs["analysis_grid"] = "500m"
    merged_reflectance.attrs["resampling_1km_to_500m"] = "nearest"

    prepared_data_vars = {
        "reflectance": merged_reflectance,
    }
    if keep_intermediate_reflectance:
        prepared_data_vars["reflectance_500m_native"] = imagery_500m.transpose("y", "x", "band")
        prepared_data_vars["reflectance_1km_on_500m"] = moderate_500m.transpose("y", "x", "band")

    prepared = xr.Dataset(
        data_vars=prepared_data_vars,
        coords={
            "x": x.values,
            "y": y.values,
            "band": merged_reflectance.coords["band"].values,
        },
        attrs=raw.attrs.copy(),
    )

    fields_1km_to_expand = [
        "solar_zenith",
        "solar_azimuth",
        "sensor_zenith",
        "sensor_azimuth",
        "qa_qf1",
        "qa_qf2",
        "qa_qf3",
        "qa_qf4",
        "qa_qf5",
        "qa_qf6",
        "qa_qf7",
        "land_water_mask",
        "num_observations_1km",
        "obscov_1km",
        "orbit_pnt",
    ]
    for variable_name in fields_1km_to_expand:
        prepared[variable_name] = _upsample_1km_dataarray_to_500m(raw[variable_name], target_x=x, target_y=y)

    for variable_name in ("iobs_res", "num_observations_500m", "obscov_500m"):
        prepared[variable_name] = _normalize_500m_dataarray(raw[variable_name])

    prepared["qa_raw_stack"] = xr.concat(
        [prepared[f"qa_qf{i}"] for i in range(1, 8)],
        dim=xr.IndexVariable("qa_flag", [f"QF{i}" for i in range(1, 8)]),
    ).transpose("y", "x", "qa_flag")

    qa_mask_ds = decode_viirs_qa_masks(
        prepared["qa_qf1"],
        prepared["qa_qf2"],
        prepared["qa_qf7"],
    )
    prepared.update(qa_mask_ds)

    mask_cloud = prepared["mask_cloud_qa"]
    mask_cloud_shadow = prepared["mask_cloud_shadow_qa"]
    if cloud_mask_source is not None:
        external_mask_ds = load_external_cloud_masks(
            cloud_mask_source,
            target_x=x,
            target_y=y,
            cloud_mask_var=cloud_mask_var,
            cloud_shadow_mask_var=cloud_shadow_mask_var,
        )
        prepared.update(external_mask_ds)
        mask_cloud = mask_cloud | prepared["mask_cloud_external"]
        mask_cloud_shadow = mask_cloud_shadow | prepared["mask_cloud_shadow_external"]

    mask_snow = prepared["mask_snow_qa"]
    mask_water_external = None
    if mask_water_using_external_file and water_mask_source is not None:
        mask_water_external = load_external_mask_on_grid(
            water_mask_source,
            target_x=x,
            target_y=y,
            variable=water_mask_var,
            name="mask_water_external",
        )
    mask_ice_external = None
    if ice_mask_source is not None:
        mask_ice_external = load_external_mask_on_grid(
            ice_mask_source,
            target_x=x,
            target_y=y,
            variable=ice_mask_var,
            name="mask_ice_external",
        )
    mask_playa_external = None
    if playa_mask_source is not None:
        mask_playa_external = load_external_mask_on_grid(
            playa_mask_source,
            target_x=x,
            target_y=y,
            variable=playa_mask_var,
            name="mask_playa_external",
        )

    mask_ds = _build_component_masks(
        prepared["reflectance"],
        prepared["sensor_zenith"],
        prepared["solar_zenith"],
        prepared["land_water_mask"],
        prepared["num_observations_1km"],
        prepared["num_observations_500m"],
        mask_cloud,
        mask_cloud_shadow,
        mask_snow,
        mask_water_external,
        mask_ice_external,
        mask_playa_external,
        water_mask_values=water_mask_values,
        mask_water_using_reflectance_qf=mask_water_using_reflectance_qf,
        mask_low_reflectance_for_inversion=mask_low_reflectance_for_inversion,
        low_reflectance_threshold=low_reflectance_threshold,
        max_sensor_zenith=max_sensor_zenith,
        max_solar_zenith=max_solar_zenith,
        min_obs_1km=min_obs_1km,
        min_obs_500m=min_obs_500m,
    )
    prepared.update(mask_ds)
    prepared = _filter_mask_outputs(prepared, write_detailed_masks=write_detailed_masks)

    prepared["reflectance"].attrs["selected_bands"] = bands
    prepared["reflectance"].attrs["band_selection_source"] = band_selection_source
    if lut_file is not None:
        prepared["reflectance"].attrs["lut_file"] = str(lut_file)

    prepared = copy_spatial_metadata(raw, prepared)
    prepared = attach_spatial_ref(
        prepared,
        x_dim="x",
        y_dim="y",
        grid_metadata=None,
        data_var_names=tuple(name for name in prepared.data_vars if name != "spatial_ref"),
    )

    log_event(
        logger,
        "prepare_viirs_scene_for_inversion",
        source_type="dataset" if isinstance(source, xr.Dataset) else "path",
        input_path=str(source) if not isinstance(source, xr.Dataset) else None,
        lut_file=str(lut_file) if lut_file is not None else None,
        selected_bands=bands,
        band_selection_source=band_selection_source,
        cloud_mask_source=str(cloud_mask_source) if isinstance(cloud_mask_source, (str, Path)) else type(cloud_mask_source).__name__ if cloud_mask_source is not None else None,
        water_mask_source=str(water_mask_source) if isinstance(water_mask_source, (str, Path)) else type(water_mask_source).__name__ if water_mask_source is not None else None,
        ice_mask_source=str(ice_mask_source) if isinstance(ice_mask_source, (str, Path)) else type(ice_mask_source).__name__ if ice_mask_source is not None else None,
        playa_mask_source=str(playa_mask_source) if isinstance(playa_mask_source, (str, Path)) else type(playa_mask_source).__name__ if playa_mask_source is not None else None,
        keep_intermediate_reflectance=keep_intermediate_reflectance,
        mask_water_using_reflectance_qf=mask_water_using_reflectance_qf,
        mask_water_using_external_file=mask_water_using_external_file,
        mask_low_reflectance_for_inversion=mask_low_reflectance_for_inversion,
        low_reflectance_threshold=low_reflectance_threshold,
        write_detailed_masks=write_detailed_masks,
        output_shape=list(prepared["reflectance"].shape),
        elapsed_seconds=round(perf_counter() - start_time, 6),
    )

    return prepared


def _filter_mask_outputs(
    prepared: xr.Dataset,
    *,
    write_detailed_masks: bool,
) -> xr.Dataset:
    if write_detailed_masks:
        return prepared

    mask_vars = [
        name
        for name in prepared.data_vars
        if name.startswith("mask_") or name == "valid_r0_mask"
    ]
    return prepared.drop_vars(mask_vars)
