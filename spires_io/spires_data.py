"""Xarray-backed SPIReS inversion input data."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from spires_io.clustering import ClusteredSpectra, Tolerance


SCENE_REQUIRED_VARIABLES = ("reflectance", "solar_zenith", "valid_inversion_mask")


@dataclass(frozen=True)
class SpiresData:
    """Container for a prepared scene plus optional background and ancillary data."""

    scene: xr.Dataset
    background: xr.DataArray | None = None
    ancillary: xr.Dataset | None = None
    cluster_defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_scene(
        cls,
        scene: xr.Dataset,
        *,
        background: xr.DataArray | None = None,
        ancillary: xr.Dataset | None = None,
        cluster_defaults: Mapping[str, Any] | None = None,
    ) -> "SpiresData":
        """Wrap an already-prepared canonical scene."""
        _validate_scene(scene)
        data = cls(
            scene=scene.copy(),
            ancillary=ancillary.copy() if ancillary is not None else None,
            cluster_defaults=dict(cluster_defaults or {}),
        )
        if background is not None:
            data = data.assign_background(background)
        return data

    @classmethod
    def from_config(cls, config_file: str) -> "SpiresData":
        """Load a single-scene config through :class:`SpiresDataLoader`."""
        from spires_io.loader import SpiresDataLoader

        return SpiresDataLoader.from_config(config_file).load()

    @property
    def target_spectra(self) -> xr.DataArray:
        """Prepared target reflectance spectra."""
        return self.scene["reflectance"]

    @property
    def solar_zenith(self) -> xr.DataArray:
        """Solar zenith angles aligned to the analysis grid."""
        return self.scene["solar_zenith"]

    @property
    def valid_mask(self) -> xr.DataArray:
        """Mask where True indicates pixels usable for inversion."""
        return self.scene["valid_inversion_mask"]

    @property
    def background_spectra(self) -> xr.DataArray | None:
        """Assigned background reflectance spectra, if present."""
        return self.background

    def assign_background(self, background: xr.DataArray) -> "SpiresData":
        """Return a new object with background reflectance assigned."""
        validated = _validate_background(self.scene, background)
        return SpiresData(
            scene=self.scene.copy(),
            background=validated.copy(),
            ancillary=self.ancillary.copy() if self.ancillary is not None else None,
            cluster_defaults=dict(self.cluster_defaults),
        )

    def assign_ancillary(self, ancillary: xr.Dataset) -> "SpiresData":
        """Return a new object with ancillary data assigned."""
        return SpiresData(
            scene=self.scene.copy(),
            background=self.background.copy() if self.background is not None else None,
            ancillary=ancillary.copy(),
            cluster_defaults=dict(self.cluster_defaults),
        )

    def assign_viewable_canopy_fraction(
        self,
        *,
        average_vertical_crown_radius: float = 4.644,
        average_horizontal_crown_radius: float = 1.72,
    ) -> "SpiresData":
        """Return a new object with viewable canopy fraction assigned."""
        if self.ancillary is None:
            raise ValueError("ancillary data must include canopy_fraction")
        if "canopy_fraction" not in self.ancillary:
            raise ValueError("ancillary data must include canopy_fraction")
        if "sensor_zenith" not in self.scene:
            raise ValueError("scene must include sensor_zenith")
        if "sensor_azimuth" not in self.scene:
            raise ValueError("scene must include sensor_azimuth")

        ancillary = self.ancillary.copy()
        canopy_fraction = _validate_ancillary_layer(
            self.scene,
            ancillary["canopy_fraction"],
            "canopy_fraction",
        )
        slope = _optional_ancillary_layer(self.scene, ancillary, "slope", default=0.0)
        aspect = _optional_ancillary_layer(self.scene, ancillary, "aspect", default=0.0)
        sensor_zenith = _validate_scene_layer(self.scene, "sensor_zenith")
        sensor_azimuth = _validate_scene_layer(self.scene, "sensor_azimuth")

        viewable_canopy_fraction = _compute_viewable_canopy_fraction(
            canopy_fraction=canopy_fraction,
            slope=slope,
            aspect=aspect,
            sensor_zenith=sensor_zenith,
            sensor_azimuth=sensor_azimuth,
            average_vertical_crown_radius=average_vertical_crown_radius,
            average_horizontal_crown_radius=average_horizontal_crown_radius,
        )
        ancillary["viewable_canopy_fraction"] = viewable_canopy_fraction

        return SpiresData(
            scene=self.scene.copy(),
            background=self.background.copy() if self.background is not None else None,
            ancillary=ancillary,
            cluster_defaults=dict(self.cluster_defaults),
        )

    def assign_mask(self, name: str, mask: xr.DataArray) -> "SpiresData":
        """Return a new object with an external inversion-exclusion mask assigned."""
        return self.assign_masks({name: mask})

    def assign_masks(self, masks: Mapping[str, xr.DataArray]) -> "SpiresData":
        """Return a new object with external inversion-exclusion masks assigned."""
        if not masks:
            return self

        scene = self.scene.copy()
        external_mask = _existing_external_mask(scene)

        for name, mask in masks.items():
            mask_name = _mask_variable_name(name)
            validated = _validate_mask(scene, mask)
            scene[mask_name] = validated
            external_mask = external_mask | validated

        external_mask = external_mask.astype(bool)
        external_mask.name = "mask_external_inversion"
        scene["mask_external_inversion"] = external_mask

        valid_mask = scene["valid_inversion_mask"] & (~external_mask)
        valid_mask.attrs.update(scene["valid_inversion_mask"].attrs)
        scene["valid_inversion_mask"] = valid_mask.astype(bool)

        return SpiresData(
            scene=scene,
            background=self.background.copy() if self.background is not None else None,
            ancillary=self.ancillary.copy() if self.ancillary is not None else None,
            cluster_defaults=dict(self.cluster_defaults),
        )

    def inversion_inputs(self) -> dict[str, xr.DataArray]:
        """Return arrays named for the spires-inversion boundary."""
        if self.background is None:
            raise ValueError("background must be assigned before building inversion inputs")

        return {
            "spectra_targets": self.target_spectra.astype("float32"),
            "spectra_backgrounds": self.background.astype("float32"),
            "obs_solar_angles": self.solar_zenith.astype("float32"),
            "valid_mask": self.valid_mask,
        }

    def to_dataset(self) -> xr.Dataset:
        """Return the prepared scene, including assigned background if available."""
        dataset = self.scene.copy()
        if self.background is not None:
            dataset["background_reflectance"] = self.background.copy()
        return dataset

    def cluster(
        self,
        *,
        features: Sequence[str] | None = None,
        label_name: str | None = None,
        valid_mask: xr.DataArray | np.ndarray | None = None,
        representative_method: str | None = None,
        reflectance_tol: "Tolerance | None" = None,
        background_tol: "Tolerance | None" = None,
        solar_zenith_tol: "Tolerance | None" = None,
    ) -> "SpiresData":
        """Return a new object with cluster labels and representatives on the scene."""
        from spires_io.clustering import cluster_spectra_block

        defaults = _cluster_defaults(self.cluster_defaults)

        cluster_valid_mask = (
            self.valid_mask
            if valid_mask is None
            else _validate_valid_mask_override(self.scene, valid_mask)
        )

        clustered = cluster_spectra_block(
            reflectance=self.target_spectra.values,
            background=None if self.background is None else self.background.values,
            solar_zenith=self.solar_zenith.values,
            features=features if features is not None else defaults["features"],
            valid_mask=cluster_valid_mask.values,
            representative_method=(
                representative_method
                if representative_method is not None
                else defaults["representative_method"]
            ),
            reflectance_tol=(
                reflectance_tol
                if reflectance_tol is not None
                else defaults["reflectance_tol"]
            ),
            background_tol=(
                background_tol
                if background_tol is not None
                else defaults["background_tol"]
            ),
            solar_zenith_tol=(
                solar_zenith_tol
                if solar_zenith_tol is not None
                else defaults["solar_zenith_tol"]
            ),
        )
        scene = _assign_cluster_outputs(
            self.scene,
            clustered,
            label_name=label_name if label_name is not None else defaults["label_name"],
        )
        return SpiresData(
            scene=scene,
            background=self.background.copy() if self.background is not None else None,
            ancillary=self.ancillary.copy() if self.ancillary is not None else None,
            cluster_defaults=dict(self.cluster_defaults),
        )


def _cluster_defaults(defaults: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "features": defaults.get("features"),
        "label_name": defaults.get("label_name", "cluster_label"),
        "representative_method": defaults.get("representative_method", "cluster_mean"),
        "reflectance_tol": defaults.get("reflectance_tol", 0.02),
        "background_tol": defaults.get("background_tol", 0.02),
        "solar_zenith_tol": defaults.get("solar_zenith_tol", 2.0),
    }


def _validate_scene(scene: xr.Dataset) -> None:
    missing = [name for name in SCENE_REQUIRED_VARIABLES if name not in scene]
    if missing:
        raise ValueError(f"prepared scene is missing required variables: {missing}")

    if scene["reflectance"].dims != ("y", "x", "band"):
        raise ValueError("scene['reflectance'] must have dims ('y', 'x', 'band')")
    if scene["solar_zenith"].dims != ("y", "x"):
        raise ValueError("scene['solar_zenith'] must have dims ('y', 'x')")
    if scene["valid_inversion_mask"].dims != ("y", "x"):
        raise ValueError("scene['valid_inversion_mask'] must have dims ('y', 'x')")


def _validate_background(scene: xr.Dataset, background: xr.DataArray) -> xr.DataArray:
    if background.dims != ("y", "x", "band"):
        raise ValueError("background must have dims ('y', 'x', 'band')")
    _require_matching_coords(scene, background, ("y", "x", "band"), "background")
    return background.astype(background.dtype, copy=False)


def _validate_mask(scene: xr.Dataset, mask: xr.DataArray) -> xr.DataArray:
    if mask.dims != ("y", "x"):
        raise ValueError("mask must have dims ('y', 'x')")
    _require_matching_coords(scene, mask, ("y", "x"), "mask")
    return mask.astype(bool)


def _validate_scene_layer(scene: xr.Dataset, name: str) -> xr.DataArray:
    layer = scene[name]
    if layer.dims != ("y", "x"):
        raise ValueError(f"scene[{name!r}] must have dims ('y', 'x')")
    _require_matching_coords(scene, layer, ("y", "x"), f"scene[{name!r}]")
    return layer.astype("float32")


def _validate_ancillary_layer(
    scene: xr.Dataset,
    layer: xr.DataArray,
    name: str,
) -> xr.DataArray:
    if layer.dims != ("y", "x"):
        raise ValueError(f"ancillary[{name!r}] must have dims ('y', 'x')")
    _require_matching_coords(scene, layer, ("y", "x"), f"ancillary[{name!r}]")
    return layer.astype("float32")


def _optional_ancillary_layer(
    scene: xr.Dataset,
    ancillary: xr.Dataset,
    name: str,
    *,
    default: float,
) -> xr.DataArray | float:
    if name not in ancillary:
        return default
    return _validate_ancillary_layer(scene, ancillary[name], name)


def _compute_viewable_canopy_fraction(
    *,
    canopy_fraction: xr.DataArray,
    slope: xr.DataArray | float,
    aspect: xr.DataArray | float,
    sensor_zenith: xr.DataArray,
    sensor_azimuth: xr.DataArray,
    average_vertical_crown_radius: float,
    average_horizontal_crown_radius: float,
) -> xr.DataArray:
    if average_vertical_crown_radius <= 0:
        raise ValueError("average_vertical_crown_radius must be > 0")
    if average_horizontal_crown_radius <= 0:
        raise ValueError("average_horizontal_crown_radius must be > 0")

    b_r = average_vertical_crown_radius / average_horizontal_crown_radius
    theta_v_prime = np.arctan(b_r * np.tan(np.deg2rad(sensor_zenith)))
    theta_s_prime = np.deg2rad(
        90.0 - np.rad2deg(np.arctan(b_r * np.tan(np.deg2rad(90.0 - slope))))
    )
    phi_v_prime = np.deg2rad(sensor_azimuth - aspect)

    exponent = np.cos(theta_s_prime) / (
        np.cos(phi_v_prime) * np.sin(theta_v_prime) * np.sin(theta_s_prime)
        + np.cos(theta_v_prime) * np.cos(theta_s_prime)
    )
    viewable_canopy_fraction = 1.0 - ((1.0 - canopy_fraction) ** exponent)
    viewable_canopy_fraction = viewable_canopy_fraction.astype("float32")
    viewable_canopy_fraction.name = "viewable_canopy_fraction"
    viewable_canopy_fraction.attrs.update(
        {
            "long_name": "viewable canopy fraction",
            "source": "canopy_fraction adjusted for terrain and sensor view geometry",
            "average_vertical_crown_radius": average_vertical_crown_radius,
            "average_horizontal_crown_radius": average_horizontal_crown_radius,
        }
    )
    return viewable_canopy_fraction


def _validate_valid_mask_override(
    scene: xr.Dataset,
    valid_mask: xr.DataArray | np.ndarray,
) -> xr.DataArray:
    if isinstance(valid_mask, xr.DataArray):
        if valid_mask.dims != ("y", "x"):
            raise ValueError("valid_mask must have dims ('y', 'x')")
        _require_matching_coords(scene, valid_mask, ("y", "x"), "valid_mask")
        return valid_mask.astype(bool)

    return xr.DataArray(
        np.asarray(valid_mask, dtype=bool),
        dims=("y", "x"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x")},
        name="valid_mask",
    )


def _assign_cluster_outputs(
    scene: xr.Dataset,
    clustered: "ClusteredSpectra",
    *,
    label_name: str,
) -> xr.Dataset:
    label_name = _validate_cluster_label_name(label_name)
    updated = scene.copy()
    cluster_coord = np.arange(clustered.n_clusters, dtype=np.int64)

    labels = np.full(updated["valid_inversion_mask"].shape, -1, dtype=np.int64).reshape(-1)
    if clustered.n_valid > 0:
        labels[clustered.valid_flat_indices] = clustered.inverse_indices
    labels = labels.reshape(updated["valid_inversion_mask"].shape)

    updated[label_name] = xr.DataArray(
        labels,
        dims=("y", "x"),
        coords={dim: updated.coords[dim].values for dim in ("y", "x")},
        name=label_name,
        attrs=_cluster_attrs(clustered),
    )
    updated["cluster_count"] = xr.DataArray(
        clustered.counts.astype(np.int64, copy=False),
        dims=("cluster",),
        coords={"cluster": cluster_coord},
        name="cluster_count",
        attrs=_cluster_attrs(clustered),
    )

    if clustered.representative_reflectance is not None:
        updated["cluster_representative_reflectance"] = xr.DataArray(
            clustered.representative_reflectance,
            dims=("cluster", "band"),
            coords={
                "cluster": cluster_coord,
                "band": updated.coords["band"].values,
            },
            name="cluster_representative_reflectance",
            attrs=_cluster_attrs(clustered),
        )
    if clustered.representative_background is not None:
        updated["cluster_representative_background"] = xr.DataArray(
            clustered.representative_background,
            dims=("cluster", "band"),
            coords={
                "cluster": cluster_coord,
                "band": updated.coords["band"].values,
            },
            name="cluster_representative_background",
            attrs=_cluster_attrs(clustered),
        )
    if clustered.representative_solar_zenith is not None:
        updated["cluster_representative_solar_zenith"] = xr.DataArray(
            clustered.representative_solar_zenith,
            dims=("cluster",),
            coords={"cluster": cluster_coord},
            name="cluster_representative_solar_zenith",
            attrs=_cluster_attrs(clustered),
        )

    return updated


def _cluster_attrs(clustered: "ClusteredSpectra") -> dict[str, str]:
    return {
        "features": ",".join(clustered.features),
        "representative_method": clustered.representative_method,
    }


def _validate_cluster_label_name(label_name: str) -> str:
    cleaned = label_name.strip()
    if not cleaned:
        raise ValueError("label_name must be non-empty")
    return cleaned


def _require_matching_coords(
    scene: xr.Dataset,
    data_array: xr.DataArray,
    dims: tuple[str, ...],
    label: str,
) -> None:
    for dim in dims:
        if dim not in data_array.coords:
            raise ValueError(f"{label} is missing coordinate {dim!r}")
        if dim not in scene.coords:
            raise ValueError(f"scene is missing coordinate {dim!r}")
        if not np.array_equal(data_array.coords[dim].values, scene.coords[dim].values):
            raise ValueError(f"{label} coordinate {dim!r} does not match the scene")


def _existing_external_mask(scene: xr.Dataset) -> xr.DataArray:
    if "mask_external_inversion" in scene:
        return scene["mask_external_inversion"].astype(bool)

    valid_mask = scene["valid_inversion_mask"]
    return xr.zeros_like(valid_mask, dtype=bool)


def _mask_variable_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("mask name must be non-empty")
    if cleaned.startswith("mask_"):
        return cleaned
    return f"mask_{cleaned}"
