"""Reader and scene-prep helpers for MODIS MOD09GA / MYD09GA products."""

from dataclasses import asdict
from datetime import datetime, timedelta
import logging
import re
from pathlib import Path
from time import perf_counter

from netCDF4 import Dataset as NetCDFDataset
import numpy as np
import xarray as xr

from spires_io.logging_utils import log_event
from spires_io.base import SceneMetadata, collect_attrs, normalize_path, read_scaled_array
from spires_io.modis.bands import MODIS_PRODUCT_TO_PLATFORM, resolve_modis_inversion_bands
from spires_io.modis.geospatial import attach_spatial_ref, copy_spatial_metadata, parse_modis_grid_metadata
from spires_io.modis.qa import decode_modis_qa_masks, load_external_cloud_masks


MODIS_FILENAME_RE = re.compile(
    r"^(?P<product>MOD09GA|MYD09GA)\.A(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?P<tile>h\d{2}v\d{2})\.(?P<collection>\d{3})\.(?P<processing>\d+)\.(?P<suffix>hdf|h5)$",
    re.IGNORECASE,
)

LOGGER = logging.getLogger(__name__)
MODIS_1KM_GRID_NAME = "MODIS_Grid_1km_2D"
MODIS_500M_GRID_NAME = "MODIS_Grid_500m_2D"
MODIS_500M_REFLECTANCE_BANDS = ("1", "2", "3", "4", "5", "6", "7")
MODIS_1KM_GEOMETRY_FIELDS = {
    "sensor_zenith": "SensorZenith_1",
    "sensor_azimuth": "SensorAzimuth_1",
    "solar_zenith": "SolarZenith_1",
    "solar_azimuth": "SolarAzimuth_1",
}
MODIS_1KM_QA_FIELDS = {
    "state_1km": "state_1km_1",
    "num_observations_1km": "num_observations_1km",
    "range_1km": "Range_1",
    "gflags_1km": "gflags_1",
    "orbit_pnt_1km": "orbit_pnt_1",
    "granule_pnt_1km": "granule_pnt_1",
}
MODIS_500M_SUPPORT_FIELDS = {
    "num_observations_500m": "num_observations_500m",
    "qc_500m": "QC_500m_1",
    "obscov_500m": "obscov_500m_1",
    "iobs_res_500m": "iobs_res_1",
    "q_scan_500m": "q_scan_1",
}


def parse_modis_surface_reflectance_filename(path: str | Path) -> SceneMetadata:
    """Parse standard MOD09GA / MYD09GA filenames into normalized metadata."""
    path = normalize_path(path)
    match = MODIS_FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized MODIS surface reflectance filename: {path.name}")

    product = match.group("product").upper()
    year = int(match.group("year"))
    doy = int(match.group("doy"))
    acquisition_date = (datetime(year, 1, 1) + timedelta(days=doy - 1)).date().isoformat()

    return SceneMetadata(
        product=product,
        platform=MODIS_PRODUCT_TO_PLATFORM[product],
        tile=match.group("tile"),
        acquisition_date=acquisition_date,
        collection=match.group("collection"),
        processing_timestamp=match.group("processing"),
        source_path=str(path),
    )


def _open_modis_netcdf(path: str | Path) -> NetCDFDataset:
    dataset = NetCDFDataset(path)
    dataset.set_auto_mask(False)
    dataset.set_auto_scale(False)
    return dataset


def _reflectance_field_name(band_name: str) -> str:
    return f"sur_refl_b{int(str(band_name)):02d}_1"


def _open_band_stack(
    dataset: NetCDFDataset,
    band_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, str]]:
    arrays = []
    units_by_band = {}
    for band_name in band_names:
        variable = dataset.variables[_reflectance_field_name(band_name)]
        arrays.append(read_scaled_array(variable))
        units_by_band[band_name] = collect_attrs(variable).get("units", "")
    stacked = np.stack(arrays, axis=-1)
    return stacked, units_by_band


def _open_scalar_fields(
    dataset: NetCDFDataset,
    field_map: dict[str, str],
    *,
    dims: tuple[str, str],
    apply_scale: bool,
    mask_fill: bool,
    mask_valid_range: bool,
) -> dict[str, xr.DataArray]:
    result = {}
    for variable_name, dataset_name in field_map.items():
        variable = dataset.variables[dataset_name]
        array = read_scaled_array(
            variable,
            apply_scale=apply_scale,
            mask_fill=mask_fill,
            mask_valid_range=mask_valid_range,
        )
        result[variable_name] = xr.DataArray(array, dims=dims, attrs=collect_attrs(variable))
    return result


def _upsample_1km_array_to_500m(array: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(array, 2, axis=0), 2, axis=1)


def _upsample_1km_dataarray_to_500m(data_array: xr.DataArray, *, target_x: xr.DataArray, target_y: xr.DataArray) -> xr.DataArray:
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
    land_water_class: xr.DataArray,
    num_observations_1km: xr.DataArray,
    num_observations_500m: xr.DataArray,
    modland_qa: xr.DataArray,
    mask_cloud: xr.DataArray,
    mask_cloud_shadow: xr.DataArray,
    mask_snow: xr.DataArray,
    *,
    water_mask_values: tuple[int, ...],
    max_sensor_zenith: float,
    max_solar_zenith: float,
    min_obs_1km: int,
    min_obs_500m: int,
    cloud_mask_policy: str,
) -> xr.Dataset:
    finite_reflectance = np.isfinite(reflectance)
    mask_invalid_reflectance = ~finite_reflectance.all(dim="band")

    mask_bad_geometry = (
        (~np.isfinite(sensor_zenith))
        | (~np.isfinite(solar_zenith))
        | (sensor_zenith > max_sensor_zenith)
        | (solar_zenith > max_solar_zenith)
    )

    mask_water = xr.zeros_like(land_water_class, dtype=bool)
    for value in water_mask_values:
        mask_water = mask_water | (land_water_class == value)

    mask_low_observation_support = (
        (num_observations_1km < min_obs_1km)
        | (num_observations_500m < min_obs_500m)
    )

    mask_bad_modland_qa = modland_qa >= 2

    mask_shape = land_water_class.shape
    mask_dims = land_water_class.dims
    mask_coords = {dim: land_water_class.coords[dim].values for dim in mask_dims}
    false_mask = xr.DataArray(np.zeros(mask_shape, dtype=bool), dims=mask_dims, coords=mask_coords)

    if cloud_mask_policy == "strict":
        mask_cloud_for_inversion = mask_cloud
        mask_cloud_shadow_for_inversion = mask_cloud_shadow
    elif cloud_mask_policy == "snow_wins":
        mask_cloud_for_inversion = mask_cloud & (~mask_snow)
        mask_cloud_shadow_for_inversion = mask_cloud_shadow & (~mask_snow)
    elif cloud_mask_policy == "ignore_cloud":
        mask_cloud_for_inversion = false_mask.copy()
        mask_cloud_shadow_for_inversion = mask_cloud_shadow
    elif cloud_mask_policy == "ignore_cloud_and_shadow":
        mask_cloud_for_inversion = false_mask.copy()
        mask_cloud_shadow_for_inversion = false_mask.copy()
    else:
        raise ValueError(
            "cloud_mask_policy must be one of "
            "'strict', 'snow_wins', 'ignore_cloud', or 'ignore_cloud_and_shadow'"
        )

    valid_inversion_mask = ~(
        mask_invalid_reflectance
        | mask_bad_geometry
        | mask_water
        | mask_low_observation_support
        | mask_bad_modland_qa
        | mask_cloud_for_inversion
        | mask_cloud_shadow_for_inversion
    )
    valid_r0_mask = ~(
        mask_invalid_reflectance
        | mask_bad_geometry
        | mask_low_observation_support
        | mask_bad_modland_qa
        | mask_cloud
        | mask_cloud_shadow
        | mask_snow
    )

    return xr.Dataset(
        data_vars={
            "mask_invalid_reflectance": mask_invalid_reflectance.astype(bool),
            "mask_bad_geometry": mask_bad_geometry.astype(bool),
            "mask_water": mask_water.astype(bool),
            "mask_low_observation_support": mask_low_observation_support.astype(bool),
            "mask_bad_modland_qa": mask_bad_modland_qa.astype(bool),
            "mask_cloud": mask_cloud.astype(bool),
            "mask_cloud_shadow": mask_cloud_shadow.astype(bool),
            "mask_snow": mask_snow.astype(bool),
            "mask_cloud_for_inversion": mask_cloud_for_inversion.astype(bool),
            "mask_cloud_shadow_for_inversion": mask_cloud_shadow_for_inversion.astype(bool),
            "valid_inversion_mask": valid_inversion_mask.astype(bool),
            "valid_r0_mask": valid_r0_mask.astype(bool),
        }
    )


def open_modis_surface_reflectance(
    path: str | Path,
    *,
    bands: tuple[str, ...] | list[str] | None = None,
    lut_file: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> xr.Dataset:
    """
    Open a single MOD09GA or MYD09GA file as a normalized xarray dataset.
    """
    start_time = perf_counter()
    logger = logger or LOGGER
    path = normalize_path(path)
    scene = parse_modis_surface_reflectance_filename(path)
    selected_bands = resolve_modis_inversion_bands(bands=bands, lut_file=lut_file)

    with _open_modis_netcdf(path) as dataset:
        grid_metadata_1km = parse_modis_grid_metadata(dataset, MODIS_1KM_GRID_NAME)
        grid_metadata_500m = parse_modis_grid_metadata(dataset, MODIS_500M_GRID_NAME)
        if grid_metadata_1km is None or grid_metadata_500m is None:
            raise ValueError(f"Could not parse MODIS grid metadata from {path.name}")

        reflectance_500m, units_500m = _open_band_stack(dataset, tuple(selected_bands))

        ds = xr.Dataset(
            data_vars={
                "reflectance_500m": xr.DataArray(
                    reflectance_500m,
                    dims=("y_500m", "x_500m", "band_500m"),
                    coords={
                        "y_500m": grid_metadata_500m.y_coords(),
                        "x_500m": grid_metadata_500m.x_coords(),
                        "band_500m": selected_bands,
                    },
                    attrs={"units_by_band": units_500m},
                ),
            },
            coords={
                "x_1km": grid_metadata_1km.x_coords(),
                "y_1km": grid_metadata_1km.y_coords(),
                "x_500m": grid_metadata_500m.x_coords(),
                "y_500m": grid_metadata_500m.y_coords(),
                "band_500m": selected_bands,
            },
            attrs=asdict(scene),
        )

        ds.update(
            _open_scalar_fields(
                dataset,
                MODIS_1KM_GEOMETRY_FIELDS,
                dims=("y_1km", "x_1km"),
                apply_scale=True,
                mask_fill=True,
                mask_valid_range=True,
            )
        )
        ds.update(
            _open_scalar_fields(
                dataset,
                MODIS_1KM_QA_FIELDS,
                dims=("y_1km", "x_1km"),
                apply_scale=False,
                mask_fill=True,
                mask_valid_range=True,
            )
        )
        ds.update(
            _open_scalar_fields(
                dataset,
                MODIS_500M_SUPPORT_FIELDS,
                dims=("y_500m", "x_500m"),
                apply_scale=False,
                mask_fill=True,
                mask_valid_range=True,
            )
        )

        ds["reflectance_500m"].attrs["long_name"] = "MODIS 500 m surface reflectance"
        ds["reflectance_500m"].attrs.update(grid_metadata_500m.to_attrs())
        ds.attrs["selected_bands"] = selected_bands
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
        "open_modis_surface_reflectance",
        input_path=str(path),
        product=scene.product,
        platform=scene.platform,
        tile=scene.tile,
        acquisition_date=scene.acquisition_date,
        selected_bands=selected_bands,
        elapsed_seconds=round(perf_counter() - start_time, 6),
    )
    return ds


def prepare_modis_scene_for_inversion(
    source: str | Path | xr.Dataset,
    *,
    bands: tuple[str, ...] | list[str] | None = None,
    lut_file: str | Path | None = None,
    logger: logging.Logger | None = None,
    cloud_mask_source: str | Path | xr.Dataset | xr.DataArray | None = None,
    cloud_mask_var: str = "mask_cloud",
    cloud_shadow_mask_var: str = "mask_cloud_shadow",
    keep_intermediate_reflectance: bool = False,
    max_sensor_zenith: float = 65.0,
    max_solar_zenith: float = 85.0,
    min_obs_1km: int = 1,
    min_obs_500m: int = 1,
    water_mask_values: tuple[int, ...] = (0, 2, 3, 4, 5, 6, 7),
    cloud_mask_policy: str = "strict",
) -> xr.Dataset:
    """Prepare a MODIS scene on a single 500 m analysis grid for inversion."""
    start_time = perf_counter()
    logger = logger or LOGGER

    if isinstance(source, xr.Dataset):
        raw = source
    else:
        raw = open_modis_surface_reflectance(source, bands=bands, lut_file=lut_file, logger=logger)

    selected_bands = resolve_modis_inversion_bands(bands=bands, lut_file=lut_file)
    x = raw["x_500m"]
    y = raw["y_500m"]

    reflectance = _normalize_500m_dataarray(raw["reflectance_500m"]).sel(band=selected_bands).transpose("y", "x", "band")
    reflectance.name = "reflectance"
    reflectance.attrs["analysis_grid"] = "500m"

    prepared_data_vars = {"reflectance": reflectance}
    if keep_intermediate_reflectance:
        prepared_data_vars["reflectance_500m_native"] = reflectance.copy()

    prepared = xr.Dataset(
        data_vars=prepared_data_vars,
        coords={"x": x.values, "y": y.values, "band": reflectance.coords["band"].values},
        attrs=raw.attrs.copy(),
    )

    fields_1km_to_expand = (
        "sensor_zenith",
        "sensor_azimuth",
        "solar_zenith",
        "solar_azimuth",
        "state_1km",
        "num_observations_1km",
        "range_1km",
        "gflags_1km",
        "orbit_pnt_1km",
        "granule_pnt_1km",
    )
    for variable_name in fields_1km_to_expand:
        prepared[variable_name] = _upsample_1km_dataarray_to_500m(raw[variable_name], target_x=x, target_y=y)

    for variable_name in ("num_observations_500m", "qc_500m", "obscov_500m", "iobs_res_500m", "q_scan_500m"):
        prepared[variable_name] = _normalize_500m_dataarray(raw[variable_name])

    qa_ds = decode_modis_qa_masks(prepared["state_1km"], prepared["qc_500m"])
    prepared.update(qa_ds)

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
        mask_cloud = prepared["mask_cloud_external"]
        mask_cloud_shadow = prepared["mask_cloud_shadow_external"]

    mask_snow = prepared["mask_snow_qa"]

    mask_ds = _build_component_masks(
        prepared["reflectance"],
        prepared["sensor_zenith"],
        prepared["solar_zenith"],
        prepared["qa_land_water_class"],
        prepared["num_observations_1km"],
        prepared["num_observations_500m"],
        prepared["qa_modland"],
        mask_cloud,
        mask_cloud_shadow,
        mask_snow,
        water_mask_values=water_mask_values,
        max_sensor_zenith=max_sensor_zenith,
        max_solar_zenith=max_solar_zenith,
        min_obs_1km=min_obs_1km,
        min_obs_500m=min_obs_500m,
        cloud_mask_policy=cloud_mask_policy,
    )
    prepared.update(mask_ds)
    prepared.attrs["cloud_mask_policy"] = cloud_mask_policy

    prepared["reflectance"].attrs["selected_bands"] = selected_bands
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
        "prepare_modis_scene_for_inversion",
        source_type="dataset" if isinstance(source, xr.Dataset) else "path",
        input_path=str(source) if not isinstance(source, xr.Dataset) else None,
        lut_file=str(lut_file) if lut_file is not None else None,
        selected_bands=selected_bands,
        cloud_mask_policy=cloud_mask_policy,
        cloud_mask_source=str(cloud_mask_source) if isinstance(cloud_mask_source, (str, Path)) else type(cloud_mask_source).__name__ if cloud_mask_source is not None else None,
        keep_intermediate_reflectance=keep_intermediate_reflectance,
        output_shape=list(prepared["reflectance"].shape),
        elapsed_seconds=round(perf_counter() - start_time, 6),
    )
    return prepared
