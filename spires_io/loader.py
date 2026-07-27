"""Orchestration layer for preparing :class:`SpiresData` inputs."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import xarray as xr
from spires_contract import (
    SpiresData,
    validate_spatial_alignment,
    validate_spires_data,
)

from spires_io.api import prepare_scene_for_inversion
from spires_io.ancillary import load_ancillary_layers
from spires_io.background import load_background_reflectance
from spires_io.configs import (
    SceneManifestItem,
    SpiresConfig,
    SpiresRunConfig,
)
from spires_io.geometry import add_illumination_geometry
import spires_io.constants as constants
from spires_io.masks import assign_inversion_exclusion_masks, load_external_mask


ScenePreparer = Callable[..., xr.Dataset]
BackgroundLoader = Callable[..., xr.DataArray | None]
AncillaryLoader = Callable[..., xr.Dataset | None]
MaskLoader = Callable[..., xr.DataArray]


def load(config_file: str | Path) -> SpiresData:
    """Load one configured scene into the shared ``SpiresData`` container."""
    return SpiresDataLoader.from_config(config_file).load()


class SpiresDataLoader:
    """Prepare canonical scenes and wrap them as :class:`SpiresData`."""

    def __init__(
        self,
        run_config: SpiresRunConfig | None = None,
        *,
        single_scene_config: SpiresConfig | None = None,
        scene_preparer: ScenePreparer = prepare_scene_for_inversion,
        background_loader: BackgroundLoader | None = None,
        ancillary_loader: AncillaryLoader = load_ancillary_layers,
        mask_loader: MaskLoader = load_external_mask,
    ) -> None:
        self.run_config = run_config
        self.single_scene_config = single_scene_config
        self._scene_preparer = scene_preparer
        self._background_loader = background_loader or load_background_reflectance
        self._ancillary_loader = ancillary_loader
        self._mask_loader = mask_loader

    @classmethod
    def from_config(
        cls,
        config_file: str | Path,
        *,
        scene_preparer: ScenePreparer = prepare_scene_for_inversion,
        background_loader: BackgroundLoader | None = None,
        ancillary_loader: AncillaryLoader = load_ancillary_layers,
        mask_loader: MaskLoader = load_external_mask,
    ) -> "SpiresDataLoader":
        """Create a loader for Brent-style single-scene JSON configs."""
        return cls(
            single_scene_config=SpiresConfig(config_file),
            scene_preparer=scene_preparer,
            background_loader=background_loader,
            ancillary_loader=ancillary_loader,
            mask_loader=mask_loader,
        )

    def load(self) -> SpiresData:
        """Load the configured single scene."""
        if self.single_scene_config is None:
            raise ValueError(
                "load() requires a single-scene config; use load_item() for manifests"
            )

        config = self.single_scene_config
        scene = self._scene_preparer(
            config.files.image_data,
            sensor=config.sensor.name,
            **_reader_kwargs_from_single_scene_config(config),
        )
        background = self._background_loader(config.files.background_image, target_scene=scene)
        ancillary = self._ancillary_loader(
            _single_scene_ancillary_sources(config),
            target_scene=scene,
        )
        scene = add_illumination_geometry(
            scene,
            ancillary,
            require_illumination=_requires_illumination_geometry(config),
        )
        return _build_spires_data(scene, background=background, ancillary=ancillary)

    def load_item(self, item: SceneManifestItem | Mapping[str, Any]) -> SpiresData:
        """Load one scene manifest item using this loader's run-wide policy."""
        if self.run_config is None:
            raise ValueError("load_item() requires a SpiresRunConfig")
        if isinstance(item, Mapping):
            item = SceneManifestItem.from_mapping(item, item_index=0)
        _validate_postprocess_ancillary(self.run_config, item)

        scene = self._scene_preparer(
            item.image_path,
            sensor=self.run_config.sensor.name,
            **_reader_kwargs_from_run_config(self.run_config),
        )
        background = self._background_loader(item.background_image, target_scene=scene)
        ancillary = self._ancillary_loader(item.ancillary, target_scene=scene)
        scene = add_illumination_geometry(
            scene,
            ancillary,
            require_illumination=_requires_illumination_geometry(self.run_config),
        )
        data = _build_spires_data(scene, background=background, ancillary=ancillary)
        return _assign_manifest_masks(data, item.masks, scene, self._mask_loader)


def _build_spires_data(
    scene: xr.Dataset,
    *,
    background: xr.DataArray | None,
    ancillary: xr.Dataset | None,
) -> SpiresData:
    data = SpiresData(
        scene=scene.copy(deep=False),
        background=(
            None if background is None else background.copy(deep=False)
        ),
        ancillary=None if ancillary is None else ancillary.copy(deep=False),
    )
    validate_spires_data(data)
    validate_spatial_alignment(data)
    return data


def load_background_image(
    path: str | Path,
    *,
    target_scene: xr.Dataset | None = None,
) -> xr.DataArray:
    """Compatibility alias for :func:`load_background_reflectance`."""
    return load_background_reflectance(path, target_scene=target_scene)


def _reader_kwargs_from_single_scene_config(config: SpiresConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = config.reader.to_reader_kwargs()
    _setdefault_mask_kwargs(kwargs, config.mask)
    kwargs["lut_file"] = config.files.lut
    if config.sensor.selected_bands is not None:
        kwargs["bands"] = list(config.sensor.selected_bands)
    if config.files.cloud_mask is not None:
        kwargs["cloud_mask_source"] = config.files.cloud_mask
    if config.files.water_mask is not None:
        kwargs["water_mask_source"] = config.files.water_mask
    if config.files.ice_mask is not None:
        kwargs["ice_mask_source"] = config.files.ice_mask
    if config.files.playa_mask is not None:
        kwargs["playa_mask_source"] = config.files.playa_mask
    return kwargs


def _reader_kwargs_from_run_config(config: SpiresRunConfig) -> dict[str, Any]:
    kwargs = config.reader.to_reader_kwargs()
    _setdefault_mask_kwargs(kwargs, config.mask)
    if config.sensor.selected_bands is not None:
        kwargs.setdefault("bands", list(config.sensor.selected_bands))
    return kwargs


def _setdefault_mask_kwargs(kwargs: dict[str, Any], mask: Any) -> None:
    for key, value in mask.to_reader_kwargs().items():
        kwargs.setdefault(key, value)


def _single_scene_ancillary_sources(config: SpiresConfig) -> dict[str, str]:
    return {
        name: path
        for name in constants.STATIC_DATA
        if (path := getattr(config.files, name)) is not None
    }


def _validate_postprocess_ancillary(
    config: SpiresRunConfig,
    item: SceneManifestItem,
) -> None:
    required = []
    if config.postprocess.apply_canopy_correction:
        required.append("canopy_fraction")
    if config.postprocess.apply_ice_adjustment:
        required.append("ice_fraction")
    if _requires_illumination_geometry(config):
        required.extend(("slope", "aspect"))

    missing = [name for name in required if item.ancillary.get(name) is None]
    if missing:
        raise ValueError(
            "manifest scene is missing ancillary input(s) required by postprocess: "
            f"{missing}"
        )


def _requires_illumination_geometry(config: SpiresConfig | SpiresRunConfig) -> bool:
    return config.postprocess.calculate_albedo or (
        config.clustering.enabled
        and "cosine_illumination" in config.clustering.features
    )


def _assign_manifest_masks(
    data: SpiresData,
    masks: Mapping[str, Any],
    scene: xr.Dataset,
    mask_loader: MaskLoader,
) -> SpiresData:
    if not masks:
        return data

    loaded_masks = {}
    for name, spec in masks.items():
        if isinstance(spec, Mapping):
            if "path" not in spec:
                raise ValueError(f"manifest mask {name!r} must include a 'path'")
            loaded_masks[name] = mask_loader(
                spec["path"],
                target_scene=scene,
                variable=spec.get("variable") or spec.get("var"),
            )
        else:
            loaded_masks[name] = mask_loader(spec, target_scene=scene)

    return assign_inversion_exclusion_masks(data, loaded_masks)
