"""QA decoding and external mask helpers for MODIS MOD09GA / MYD09GA scenes."""

from pathlib import Path

import numpy as np
import xarray as xr


def _extract_bits(values: xr.DataArray, start_bit: int, width: int = 1, *, dtype=np.uint32) -> xr.DataArray:
    """Extract a bitfield from an unsigned integer QA array."""
    mask = (1 << width) - 1
    data = (values.fillna(0).astype(dtype) >> start_bit) & mask
    coords = {dim: values.coords[dim].values for dim in values.dims}
    return xr.DataArray(data.astype(np.uint8), dims=values.dims, coords=coords)


def decode_modis_qa_masks(
    state_1km: xr.DataArray,
    qc_500m: xr.DataArray,
) -> xr.Dataset:
    """
    Decode core MODIS QA masks used for inversion and R0 workflows.

    Current policy:
    - cloud: cloud-state cloudy/mixed, internal cloud flag, or adjacent-to-cloud
    - cloud shadow: native cloud shadow bit
    - snow: MOD35 snow/ice bit or internal snow bit
    - water class: native land/water 3-bit class
    - band quality: per-band QC and MODLAND QA summary
    """
    cloud_state = _extract_bits(state_1km, start_bit=0, width=2, dtype=np.uint16)
    cloud_shadow = _extract_bits(state_1km, start_bit=2, dtype=np.uint16).astype(bool)
    land_water = _extract_bits(state_1km, start_bit=3, width=3, dtype=np.uint16)
    aerosol_quantity = _extract_bits(state_1km, start_bit=6, width=2, dtype=np.uint16)
    cirrus = _extract_bits(state_1km, start_bit=8, width=2, dtype=np.uint16)
    internal_cloud = _extract_bits(state_1km, start_bit=10, dtype=np.uint16).astype(bool)
    mod35_snow_ice = _extract_bits(state_1km, start_bit=12, dtype=np.uint16).astype(bool)
    adjacent_to_cloud = _extract_bits(state_1km, start_bit=13, dtype=np.uint16).astype(bool)
    internal_snow = _extract_bits(state_1km, start_bit=15, dtype=np.uint16).astype(bool)

    modland_qa = _extract_bits(qc_500m, start_bit=0, width=2, dtype=np.uint32)
    band_quality = []
    for band_index in range(1, 8):
        start_bit = 2 + (band_index - 1) * 4
        band_quality.append(_extract_bits(qc_500m, start_bit=start_bit, width=4, dtype=np.uint32))

    mask_cloud = (
        (cloud_state == 1)
        | (cloud_state == 2)
        | internal_cloud
        | adjacent_to_cloud
    ).astype(bool)
    mask_cloud_shadow = cloud_shadow.astype(bool)
    mask_snow = (mod35_snow_ice | internal_snow).astype(bool)
    mask_high_cirrus = (cirrus == 3).astype(bool)

    data_vars = {
        "qa_cloud_state": cloud_state,
        "qa_land_water_class": land_water,
        "qa_aerosol_quantity": aerosol_quantity,
        "qa_cirrus_class": cirrus,
        "qa_internal_cloud_flag": internal_cloud,
        "qa_cloud_shadow_flag": cloud_shadow,
        "qa_mod35_snow_ice_flag": mod35_snow_ice,
        "qa_adjacent_to_cloud_flag": adjacent_to_cloud,
        "qa_internal_snow_flag": internal_snow,
        "qa_modland": modland_qa,
        "mask_cloud_qa": mask_cloud,
        "mask_cloud_shadow_qa": mask_cloud_shadow,
        "mask_snow_qa": mask_snow,
        "mask_high_cirrus_qa": mask_high_cirrus,
    }
    for index, quality in enumerate(band_quality, start=1):
        data_vars[f"qa_band{index}_quality"] = quality

    return xr.Dataset(data_vars=data_vars)


def _false_mask_like(target_x: xr.DataArray, target_y: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((target_y.size, target_x.size), dtype=bool),
        dims=("y", "x"),
        coords={"y": target_y.values, "x": target_x.values},
    )


def _normalize_external_mask_dataarray(
    data_array: xr.DataArray,
    *,
    target_x: xr.DataArray,
    target_y: xr.DataArray,
) -> xr.DataArray:
    rename_map = {}
    if "y_500m" in data_array.dims:
        rename_map["y_500m"] = "y"
    if "x_500m" in data_array.dims:
        rename_map["x_500m"] = "x"
    normalized = data_array.rename(rename_map)

    if normalized.dims != ("y", "x"):
        raise ValueError(
            f"External mask must have dims ('y', 'x') or ('y_500m', 'x_500m'); got {normalized.dims}"
        )

    normalized = normalized.assign_coords(y=target_y.values, x=target_x.values)
    return normalized.astype(bool)


def load_external_cloud_masks(
    source: str | Path | xr.Dataset | xr.DataArray,
    *,
    target_x: xr.DataArray,
    target_y: xr.DataArray,
    cloud_mask_var: str = "mask_cloud",
    cloud_shadow_mask_var: str = "mask_cloud_shadow",
) -> xr.Dataset:
    """
    Load external cloud and cloud-shadow masks on the prepared 500 m grid.

    If a DataArray is provided, it is treated as the cloud mask and the cloud
    shadow mask defaults to all-False.
    """
    close_dataset = None
    if isinstance(source, xr.DataArray):
        dataset = xr.Dataset({cloud_mask_var: source})
    elif isinstance(source, xr.Dataset):
        dataset = source
    else:
        source = Path(source)
        try:
            dataset = xr.open_dataset(source)
            close_dataset = dataset
        except ValueError:
            data_array = xr.open_dataarray(source)
            close_dataset = data_array
            dataset = xr.Dataset({cloud_mask_var: data_array})

    try:
        if cloud_mask_var not in dataset:
            raise ValueError(f"External cloud mask source does not contain variable {cloud_mask_var!r}")

        mask_cloud = _normalize_external_mask_dataarray(
            dataset[cloud_mask_var],
            target_x=target_x,
            target_y=target_y,
        )
        if cloud_shadow_mask_var in dataset:
            mask_cloud_shadow = _normalize_external_mask_dataarray(
                dataset[cloud_shadow_mask_var],
                target_x=target_x,
                target_y=target_y,
            )
        else:
            mask_cloud_shadow = _false_mask_like(target_x, target_y)

        return xr.Dataset(
            data_vars={
                "mask_cloud_external": mask_cloud,
                "mask_cloud_shadow_external": mask_cloud_shadow,
            }
        )
    finally:
        if close_dataset is not None:
            close_dataset.close()
