"""Orchestration layer for preparing :class:`SpiresData` inputs."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import xarray as xr

from spires_io.api import prepare_scene_for_inversion
from spires_io.configs import (
    SceneManifestItem,
    SpiresConfig,
    SpiresRunConfig,
)
from spires_io.spires_data import SpiresData


ScenePreparer = Callable[..., xr.Dataset]
BackgroundLoader = Callable[[str], xr.DataArray | None]


class SpiresDataLoader:
    """Prepare canonical scenes and wrap them as :class:`SpiresData`."""

    def __init__(
        self,
        run_config: SpiresRunConfig | None = None,
        *,
        single_scene_config: SpiresConfig | None = None,
        scene_preparer: ScenePreparer = prepare_scene_for_inversion,
        background_loader: BackgroundLoader | None = None,
    ) -> None:
        self.run_config = run_config
        self.single_scene_config = single_scene_config
        self._scene_preparer = scene_preparer
        self._background_loader = background_loader or load_background_image

    @classmethod
    def from_config(
        cls,
        config_file: str | Path,
        *,
        scene_preparer: ScenePreparer = prepare_scene_for_inversion,
        background_loader: BackgroundLoader | None = None,
    ) -> "SpiresDataLoader":
        """Create a loader for Brent-style single-scene JSON configs."""
        return cls(
            single_scene_config=SpiresConfig(config_file),
            scene_preparer=scene_preparer,
            background_loader=background_loader,
        )

    def load(self) -> SpiresData:
        """Load the configured single scene."""
        if self.single_scene_config is None:
            raise ValueError("load() requires a single-scene config; use load_item() for manifests")

        config = self.single_scene_config
        scene = self._scene_preparer(
            config.files.image_data,
            sensor=config.sensor.name,
            **_reader_kwargs_from_single_scene_config(config),
        )
        background = self._background_loader(config.files.background_image)
        return SpiresData.from_scene(scene, background=background)

    def load_item(self, item: SceneManifestItem | Mapping[str, Any]) -> SpiresData:
        """Load one scene manifest item using this loader's run-wide policy."""
        if self.run_config is None:
            raise ValueError("load_item() requires a SpiresRunConfig")
        if isinstance(item, Mapping):
            item = SceneManifestItem.from_mapping(item, item_index=0)

        scene = self._scene_preparer(
            item.image_path,
            sensor=self.run_config.sensor.name,
            **_reader_kwargs_from_run_config(self.run_config),
        )
        background = self._background_loader(item.background_image)
        return SpiresData.from_scene(scene, background=background)


def load_background_image(path: str) -> xr.DataArray:
    """Load a background image in the minimal xarray-native form supported in v1."""
    suffix = Path(path).suffix.lower()
    if suffix in {".nc", ".cdf", ".netcdf"}:
        try:
            return xr.open_dataarray(path)
        except ValueError:
            dataset = xr.open_dataset(path)
            if "reflectance" in dataset:
                return dataset["reflectance"]
            data_vars = list(dataset.data_vars)
            if len(data_vars) == 1:
                return dataset[data_vars[0]]
            raise ValueError(
                "background NetCDF datasets must contain a 'reflectance' variable "
                "or exactly one data variable"
            )

    raise NotImplementedError(
        "background loading for non-xarray files is deferred to the background "
        "and ancillary loading step"
    )


def _reader_kwargs_from_single_scene_config(config: SpiresConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "lut_file": config.files.lut,
        "max_sensor_zenith": config.option.max_sensor_zenith,
        "max_solar_zenith": config.option.max_solar_zenith,
    }
    if config.sensor.selected_bands is not None:
        kwargs["bands"] = list(config.sensor.selected_bands)
    if config.files.cloud_mask is not None:
        kwargs["cloud_mask_source"] = config.files.cloud_mask
    return kwargs


def _reader_kwargs_from_run_config(config: SpiresRunConfig) -> dict[str, Any]:
    kwargs = dict(config.reader_options)
    kwargs.update(config.mask_policy)
    if config.sensor.selected_bands is not None:
        kwargs.setdefault("bands", list(config.sensor.selected_bands))
    return kwargs
