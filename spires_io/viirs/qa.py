"""QA decoding and external mask helpers for VIIRS surface reflectance scenes."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from spires_io.viirs.bands import VIIRS_ANALYSIS_BANDS, normalize_viirs_band_names


# VNP09/VJ109/VJ209 QF3 and QF4 use one Boolean bit per reflective band
# to report unusable sensor-data-record (SDR) input.
VIIRS_BAD_SDR_INPUT_BITS: Mapping[str, tuple[str, int]] = {
    "I1": ("qf4", 1),
    "I2": ("qf4", 2),
    "I3": ("qf4", 3),
    "M1": ("qf3", 0),
    "M2": ("qf3", 1),
    "M3": ("qf3", 2),
    "M4": ("qf3", 3),
    "M5": ("qf3", 4),
    "M7": ("qf3", 5),
    "M8": ("qf3", 6),
    "M10": ("qf3", 7),
    "M11": ("qf4", 0),
}


def _extract_bits(values: xr.DataArray, start_bit: int, width: int = 1) -> xr.DataArray:
    """Extract a bitfield from an unsigned integer QA byte array."""
    mask = (1 << width) - 1
    data = (values.astype(np.uint16) >> start_bit) & mask
    coords = {dim: values.coords[dim].values for dim in values.dims}
    return xr.DataArray(data.astype(np.uint8), dims=values.dims, coords=coords)


def _decode_per_band_flags(
    qa_bytes: Mapping[str, xr.DataArray],
    bit_mapping: Mapping[str, tuple[str, int]],
) -> xr.DataArray:
    """Decode a per-band QF mapping on a separate all-sensor ``qa_band`` axis."""
    decoded = []
    for band in VIIRS_ANALYSIS_BANDS:
        qf_name, bit = bit_mapping[band]
        decoded.append(_extract_bits(qa_bytes[qf_name], start_bit=bit).astype(bool))

    spatial_dims = qa_bytes["qf3"].dims
    return xr.concat(
        decoded,
        dim=xr.IndexVariable("qa_band", list(VIIRS_ANALYSIS_BANDS)),
    ).transpose(*spatial_dims, "qa_band")


def decode_viirs_qa_masks(
    qa_qf1: xr.DataArray,
    qa_qf2: xr.DataArray,
    qa_qf3: xr.DataArray,
    qa_qf4: xr.DataArray,
    qa_qf5: xr.DataArray,
    qa_qf7: xr.DataArray,
    *,
    selected_bands: Sequence[str] | None = None,
) -> xr.Dataset:
    """
    Decode the VIIRS QA fields needed for inversion and R0 masks.

    Current policy:
    - cloud: probably cloudy, confidently cloudy, thin cirrus, or adjacent-to-cloud
    - cloud shadow: native shadow bit
    - snow: native snow/ice or snow-present flags
    - poor reflectance quality: any selected band has bad SDR input, or a
      required atmospheric-correction input is missing or invalid

    QF4 bit 4 (AOT quality) and the per-band overall surface-reflectance
    quality bits in QF5-QF6 are intentionally not inversion exclusions.
    Those retrieval-quality flags are systematically set over snow even when
    the reflectance values remain finite and suitable for snow inversion.
    """
    selected = normalize_viirs_band_names(
        list(VIIRS_ANALYSIS_BANDS) if selected_bands is None else list(selected_bands)
    )
    qa_bytes = {
        "qf3": qa_qf3,
        "qf4": qa_qf4,
    }

    cloud_confidence = _extract_bits(qa_qf1, start_bit=2, width=2)

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

    bad_sdr_input = _decode_per_band_flags(qa_bytes, VIIRS_BAD_SDR_INPUT_BITS)
    selected_bad_sdr_input = bad_sdr_input.sel(qa_band=selected).any(dim="qa_band")

    aot_missing = _extract_bits(qa_qf4, start_bit=5).astype(bool)
    land_aerosol_model_invalid = _extract_bits(qa_qf4, start_bit=6).astype(bool)
    precipitable_water_missing = _extract_bits(qa_qf4, start_bit=7).astype(bool)
    ozone_missing = _extract_bits(qa_qf5, start_bit=0).astype(bool)
    surface_pressure_missing = _extract_bits(qa_qf5, start_bit=1).astype(bool)
    atmospheric_correction_inputs_bad = (
        aot_missing
        | land_aerosol_model_invalid
        | precipitable_water_missing
        | ozone_missing
        | surface_pressure_missing
    ).astype(bool)
    mask_bad_surface_reflectance_quality = (
        selected_bad_sdr_input
        | atmospheric_correction_inputs_bad
    ).astype(bool)

    return xr.Dataset(
        data_vars={
            "mask_cloud_qa": mask_cloud,
            "mask_cloud_shadow_qa": mask_cloud_shadow,
            "mask_snow_qa": mask_snow,
            "mask_bad_surface_reflectance_quality": mask_bad_surface_reflectance_quality,
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
