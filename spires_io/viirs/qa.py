"""QA decoding and external mask helpers for VIIRS surface reflectance scenes."""

from pathlib import Path

import numpy as np
import xarray as xr


def _extract_bits(values: xr.DataArray, start_bit: int, width: int = 1) -> xr.DataArray:
    """Extract a bitfield from an unsigned integer QA byte array."""
    mask = (1 << width) - 1
    data = (values.astype(np.uint16) >> start_bit) & mask
    coords = {dim: values.coords[dim].values for dim in values.dims}
    return xr.DataArray(data.astype(np.uint8), dims=values.dims, coords=coords)


def decode_viirs_qa_masks(
    qa_qf1: xr.DataArray,
    qa_qf2: xr.DataArray,
    qa_qf7: xr.DataArray,
) -> xr.Dataset:
    """
    Decode core VIIRS QA masks used for inversion and R0 workflows.

    Current policy:
    - cloud: probably cloudy, confidently cloudy, thin cirrus, or adjacent-to-cloud
    - cloud shadow: native shadow bit
    - snow: native snow/ice or snow-present flags
    """
    cloud_confidence = _extract_bits(qa_qf1, start_bit=2, width=2)
    cloud_mask_quality = _extract_bits(qa_qf1, start_bit=0, width=2)

    qf2_shadow = _extract_bits(qa_qf2, start_bit=3).astype(bool)
    qf2_snow_ice = _extract_bits(qa_qf2, start_bit=5).astype(bool)
    qf2_thin_cirrus_reflective = _extract_bits(qa_qf2, start_bit=6).astype(bool)
    qf2_thin_cirrus_emissive = _extract_bits(qa_qf2, start_bit=7).astype(bool)

    qf7_thin_cirrus = _extract_bits(qa_qf7, start_bit=4).astype(bool)
    qf7_adjacent_to_cloud = _extract_bits(qa_qf7, start_bit=1).astype(bool)
    qf7_snow_present = _extract_bits(qa_qf7, start_bit=0).astype(bool)

    mask_cloud = (
        (cloud_confidence >= 2)
        | qf2_thin_cirrus_reflective
        | qf2_thin_cirrus_emissive
        | qf7_thin_cirrus
        | qf7_adjacent_to_cloud
    ).astype(bool)
    mask_cloud_shadow = qf2_shadow.astype(bool)
    mask_snow = (qf2_snow_ice | qf7_snow_present).astype(bool)

    return xr.Dataset(
        data_vars={
            "qa_cloud_confidence": cloud_confidence,
            "qa_cloud_mask_quality": cloud_mask_quality,
            "qa_thin_cirrus_reflective": qf2_thin_cirrus_reflective,
            "qa_thin_cirrus_emissive": qf2_thin_cirrus_emissive,
            "qa_thin_cirrus_flag": qf7_thin_cirrus,
            "qa_adjacent_to_cloud": qf7_adjacent_to_cloud,
            "qa_shadow_flag": qf2_shadow,
            "qa_snow_ice_flag": qf2_snow_ice,
            "qa_snow_present_flag": qf7_snow_present,
            "mask_cloud_qa": mask_cloud,
            "mask_cloud_shadow_qa": mask_cloud_shadow,
            "mask_snow_qa": mask_snow,
        }
    )


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
        raise ValueError(f"External mask must have dims ('y', 'x') or ('y_500m', 'x_500m'); got {normalized.dims}")

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

        mask_cloud = _normalize_external_mask_dataarray(dataset[cloud_mask_var], target_x=target_x, target_y=target_y)
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
