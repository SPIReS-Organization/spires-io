"""Derived solar and terrain illumination geometry."""

from __future__ import annotations

import numpy as np
import xarray as xr


def derive_illumination_geometry(
    solar_zenith: xr.DataArray,
    *,
    solar_azimuth: xr.DataArray | None = None,
    slope: xr.DataArray | None = None,
    aspect: xr.DataArray | None = None,
) -> xr.Dataset:
    """Derive solar-zenith and optional local-illumination cosines.

    All source angles are degrees. Solar azimuth and terrain aspect are degrees
    clockwise from north. ``cosine_illumination`` describes local incidence on
    a sloped surface and does not include shadows cast by surrounding terrain.
    """
    if not isinstance(solar_zenith, xr.DataArray):
        raise TypeError("solar_zenith must be an xarray.DataArray")

    cosine_solar_zenith = np.cos(np.deg2rad(solar_zenith))
    cosine_solar_zenith = cosine_solar_zenith.astype("float32").rename(
        "cosine_solar_zenith"
    )
    cosine_solar_zenith.attrs = {
        "long_name": "Cosine of the solar zenith angle",
        "units": "1",
        "formula": "cos(deg2rad(solar_zenith))",
        "source_angle_units": "degrees",
    }

    supplied = {
        "solar_azimuth": solar_azimuth,
        "slope": slope,
        "aspect": aspect,
    }
    if all(value is None for value in supplied.values()):
        return xr.Dataset({"cosine_solar_zenith": cosine_solar_zenith})
    missing = [name for name, value in supplied.items() if value is None]
    if missing:
        raise ValueError(
            "solar_azimuth, slope, and aspect are all required to derive "
            f"cosine_illumination; missing {missing}"
        )

    geometry = {
        name: _prepare_geometry_array(value, solar_zenith, name=name)
        for name, value in supplied.items()
    }
    normalized_azimuth = xr.where(
        np.isfinite(geometry["solar_azimuth"]),
        geometry["solar_azimuth"] % 360.0,
        np.nan,
    )
    slope_values = geometry["slope"].where(
        (geometry["slope"] >= 0.0) & (geometry["slope"] <= 90.0)
    )
    aspect_values = geometry["aspect"].where(
        (geometry["aspect"] >= 0.0) & (geometry["aspect"] < 360.0)
    )
    zenith_rad = np.deg2rad(solar_zenith)
    slope_rad = np.deg2rad(slope_values)
    relative_azimuth_rad = np.deg2rad(normalized_azimuth - aspect_values)
    cosine_illumination = (
        np.sin(zenith_rad)
        * np.sin(slope_rad)
        * np.cos(relative_azimuth_rad)
        + np.cos(zenith_rad) * np.cos(slope_rad)
    ).clip(min=0.0, max=1.0)
    cosine_illumination = cosine_illumination.transpose(*solar_zenith.dims)
    cosine_illumination = cosine_illumination.astype("float32").rename(
        "cosine_illumination"
    )
    cosine_illumination.attrs = {
        "long_name": "Cosine of local solar incidence on terrain",
        "units": "1",
        "formula": (
            "sin(solar_zenith)*sin(slope)*cos(solar_azimuth-aspect) + "
            "cos(solar_zenith)*cos(slope); angles converted from degrees"
        ),
        "azimuth_convention": "degrees clockwise from north in [0, 360)",
        "solar_azimuth_normalization": "finite values modulo 360",
        "terrain_shadow_correction": "none",
        "zero_interpretation": "no direct local illumination",
    }
    return xr.Dataset(
        {
            "cosine_solar_zenith": cosine_solar_zenith,
            "cosine_illumination": cosine_illumination,
        }
    )


def add_illumination_geometry(
    scene: xr.Dataset,
    ancillary: xr.Dataset | None = None,
    *,
    require_illumination: bool = False,
) -> xr.Dataset:
    """Return a scene copy containing all derivable illumination geometry."""
    if not isinstance(scene, xr.Dataset):
        raise TypeError("scene must be an xarray.Dataset")
    if "solar_zenith" not in scene:
        if require_illumination:
            raise ValueError(
                "cosine_illumination is required but scene solar_zenith is unavailable"
            )
        return scene.copy()

    sources = {
        "solar_azimuth": scene.get("solar_azimuth"),
        "slope": None if ancillary is None else ancillary.get("slope"),
        "aspect": None if ancillary is None else ancillary.get("aspect"),
    }
    missing = [name for name, value in sources.items() if value is None]
    if missing and require_illumination:
        raise ValueError(
            "cosine_illumination is required but cannot be derived; missing "
            f"geometry input(s): {missing}"
        )

    derived = derive_illumination_geometry(
        scene["solar_zenith"],
        **({} if missing else sources),
    )
    updated = scene.copy()
    for name, values in derived.data_vars.items():
        updated[name] = values
    return updated


def _prepare_geometry_array(
    data: xr.DataArray | None,
    target: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray")
    if data.dims != target.dims:
        raise ValueError(f"{name} must have dimensions {target.dims}; got {data.dims}")
    for dim in target.dims:
        if dim not in data.coords or not data.coords[dim].equals(target.coords[dim]):
            raise ValueError(f"{name} coordinate {dim!r} does not match solar_zenith")
    return data
