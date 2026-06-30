from collections.abc import Mapping, Sequence
import numpy as np
import importlib.resources
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, List

import spires_io.constants as constants


@dataclass(init=False)
class FilesConfig:
    image_data: str
    background_image: str
    lut: str
    cloud_mask: Optional[str] = None
    dem: Optional[str] = None
    slope: Optional[str] = None
    aspect: Optional[str] = None
    skyview: Optional[str] = None
    canopy_fraction: Optional[str] = None

    def __init__(
        self,
        image_data: str,
        lut: str,
        background_image: Optional[str] = None,
        snowfree_image: Optional[str] = None,
        cloud_mask: Optional[str] = None,
        dem: Optional[str] = None,
        slope: Optional[str] = None,
        aspect: Optional[str] = None,
        skyview: Optional[str] = None,
        canopy_fraction: Optional[str] = None,
    ) -> None:
        if background_image is not None and snowfree_image is not None:
            raise ValueError(
                "Use only one of files.background_image or deprecated "
                "files.snowfree_image"
            )
        if background_image is None and snowfree_image is not None:
            warnings.warn(
                "files.snowfree_image is deprecated; use files.background_image",
                FutureWarning,
                stacklevel=2,
            )
            background_image = snowfree_image

        self.image_data = image_data
        self.background_image = background_image
        self.lut = lut
        self.cloud_mask = cloud_mask
        self.dem = dem
        self.slope = slope
        self.aspect = aspect
        self.skyview = skyview
        self.canopy_fraction = canopy_fraction
        self.__post_init__()

    def __post_init__(self) -> None:
        for f in ["image_data", "background_image", "lut"]:
            path = getattr(self, f)
            if path is None:
                raise ValueError(
                    "The following data are required: "
                    "['image_data', 'background_image', 'lut']"
                )

            path = Path(path)

            if path.is_file() and path.suffix.lower() not in constants.VALID_EXTENSIONS:
                raise ValueError(f"{path} has invalid file type: {path.suffix}")

    @property
    def snowfree_image(self) -> str:
        """Deprecated compatibility alias for background_image."""
        return self.background_image


@dataclass
class SensorConfig:
    name: str
    product_version: Optional[int] = None
    selected_bands: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.name.lower() not in constants.SUPPORTED_SENSORS:
            raise ValueError(
                f"Sensor name '{self.name}' not in supported: {constants.SUPPORTED_SENSORS}"
            )

        self.resolution = constants.DEFAULT_RESOLUTION.get(self.name.lower())

        meta = constants.SENSORS_META.get(self.name)
        self.band_names_full = meta["bands"]
        self.wavelength_full = meta["wavelength"]

        # Allows user to specify which bands to use
        if self.selected_bands is not None:
            # This is hacky, but to allow VIIRS+MODIS etc to have band names that are string and
            # hyperspectral to be band indicies which may be easier to use. TODO
            if isinstance(self.band_names_full[0], (str, np.str_)):
                mask = np.isin(
                    self.band_names_full.astype(str),
                    np.array(self.selected_bands).astype(str),
                )
            else:
                mask = np.isin(
                    self.band_names_full.astype(float),
                    np.array(self.selected_bands).astype(float),
                )
            if not np.any(mask):
                raise ValueError(f"No bands matched selection: {self.selected_bands}")

            self.band_names = self.band_names_full[mask]
            self.wavelength = self.wavelength_full[mask]

        # Search to see if we need to apply the topographic correction for hooking
        self.apply_topo_correction = meta.get("topo_map", {}).get(
            self.product_version, False
        )


@dataclass
class OptionsConfig:
    """Temporary catch-all for run options not yet promoted to a named section."""

    cpu_cores: int = 1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ignore_topography_correction: bool = True
    atmosphere_aod: float = 0.10
    atmosphere_watervapor_gcm2: float = 0.5

    def __post_init__(self) -> None:
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be >= 1")

        self.atmosphere_aod = max(
            constants.MIN_AOD, min(self.atmosphere_aod, constants.MAX_AOD)
        )

        self.atmosphere_watervapor_gcm2 = max(
            constants.MIN_H2O, min(self.atmosphere_watervapor_gcm2, constants.MAX_H2O)
        )


@dataclass(init=False)
class ReaderConfig:
    max_sensor_zenith: float = 65.0
    max_solar_zenith: float = 85.0
    cloud_mask_policy: str = "strict"
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        max_sensor_zenith: float = 65.0,
        max_solar_zenith: float = 85.0,
        cloud_mask_policy: str = "strict",
        extra: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.max_sensor_zenith = max_sensor_zenith
        self.max_solar_zenith = max_solar_zenith
        self.cloud_mask_policy = cloud_mask_policy
        self.extra = dict(extra or {})
        self.extra.update(kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        self.max_sensor_zenith = max(
            constants.MIN_ZENITH, min(self.max_sensor_zenith, constants.MAX_ZENITH)
        )

        self.max_solar_zenith = max(
            constants.MIN_ZENITH, min(self.max_solar_zenith, constants.MAX_ZENITH)
        )

        valid_policies = {
            "strict",
            "snow_wins",
            "ignore_cloud",
            "ignore_cloud_and_shadow",
        }
        if self.cloud_mask_policy not in valid_policies:
            raise ValueError(
                "cloud_mask_policy must be one of "
                f"{sorted(valid_policies)}; got {self.cloud_mask_policy!r}"
            )

    def to_reader_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.extra)
        kwargs.setdefault("max_sensor_zenith", self.max_sensor_zenith)
        kwargs.setdefault("max_solar_zenith", self.max_solar_zenith)
        kwargs.setdefault("cloud_mask_policy", self.cloud_mask_policy)
        return kwargs


@dataclass
class SpatialConfig:
    target_bbox: Optional[list] = None
    target_crs: Optional[str] = None
    resampling_method: Optional[str] = "bilinear"
    use_custom_sensor_resolution: bool = False

    def __post_init__(self) -> None:
        if self.resampling_method is not None:
            if self.resampling_method.lower() not in [
                "average",
                "bilinear",
                "cubic",
                "med",
                "nearest",
            ]:
                raise ValueError(f"Invalid resampling method: {self.resampling_method}")


@dataclass
class CanopyConfig:
    viewable_fraction: bool = False
    average_vertical_crown_radius: float = 4.644
    average_horizontal_crown_radius: float = 1.72

    def __post_init__(self) -> None:
        if self.average_vertical_crown_radius <= 0:
            raise ValueError("average_vertical_crown_radius must be > 0")
        if self.average_horizontal_crown_radius <= 0:
            raise ValueError("average_horizontal_crown_radius must be > 0")


@dataclass
class ClusterConfig:
    enabled: bool = False
    features: Sequence[str] = ("reflectance", "background", "solar_zenith")
    label_name: str = "cluster_label"
    representative_method: str = "cluster_mean"
    reflectance_tol: float | list[float] = 0.02
    background_tol: float | list[float] = 0.02
    solar_zenith_tol: float | list[float] = 2.0
    ignore_cloudmask: bool = False

    def __post_init__(self) -> None:
        from spires_io.clustering import SUPPORTED_CLUSTER_FEATURES

        if self.features is None:
            raise ValueError("clustering.features must list at least one feature")
        normalized_features = tuple(str(feature).lower() for feature in self.features)
        if not normalized_features:
            raise ValueError("clustering.features must list at least one feature")
        unknown = sorted(set(normalized_features) - SUPPORTED_CLUSTER_FEATURES)
        if unknown:
            raise ValueError(
                "unsupported clustering feature(s): "
                f"{unknown}; supported features are {sorted(SUPPORTED_CLUSTER_FEATURES)}"
            )
        if len(set(normalized_features)) != len(normalized_features):
            raise ValueError("clustering.features must not contain duplicates")
        self.features = normalized_features

        cleaned_label = self.label_name.strip()
        if not cleaned_label:
            raise ValueError("clustering.label_name must be non-empty")
        self.label_name = cleaned_label

        method = self.representative_method.lower()
        method = {"group_mean": "cluster_mean"}.get(method, method)
        if method not in {"cluster_mean", "first_pixel"}:
            raise ValueError(
                "clustering.representative_method must be one of "
                "{'cluster_mean', 'first_pixel'}"
            )
        self.representative_method = method

        for name in ("reflectance_tol", "background_tol", "solar_zenith_tol"):
            _validate_positive_tolerance(getattr(self, name), f"clustering.{name}")

    def to_cluster_kwargs(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "label_name": self.label_name,
            "representative_method": self.representative_method,
            "reflectance_tol": self.reflectance_tol,
            "background_tol": self.background_tol,
            "solar_zenith_tol": self.solar_zenith_tol,
        }


def _validate_positive_tolerance(value: float | list[float], name: str) -> None:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1:
        raise ValueError(f"{name} must be a scalar or 1D array")
    if np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive")


@dataclass
class LookUpTableConfig:
    reflectance: Optional[str] = "reflectance"
    grain_radius: Optional[str] = "grain_radius"
    pollutant: Optional[str] = "dust"
    liquid_water_fraction: Optional[str] = None
    solar_zenith: Optional[str] = "solar_zenith"


@dataclass
class InversionConfig:
    nlopt_algorithm: str = "NLOPT_LN_NELDERMEAD"
    softmax_fractional_covers: bool = True
    max_eval: int = 100
    ftol_abs: float = 1e-6
    xtol_abs: float = 1e-6

    def __post_init__(self) -> None:
        if self.nlopt_algorithm not in ["NLOPT_LN_NELDERMEAD", "NLOPT_LN_COBYLA"]:
            raise ValueError("Unsupported algorithm.")

        if (
            not self.softmax_fractional_covers
            and self.nlopt_algorithm == "NLOPT_LN_NELDERMEAD"
        ):
            raise ValueError("Algorithm requires softmax_fractional_covers=True")


@dataclass(frozen=True)
class SpiresRunConfig:
    """Batch/run-wide policy shared by one or more scene manifest items."""

    sensor: SensorConfig
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    inversion: InversionConfig = field(default_factory=InversionConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    canopy: CanopyConfig = field(default_factory=CanopyConfig)
    output_policy: dict[str, Any] = field(default_factory=dict)
    ancillary_paths: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, config_file: str | Path) -> "SpiresRunConfig":
        """Read a JSON run config file."""
        data = _load_json_object(config_file, "run config")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SpiresRunConfig":
        """Build a run config from a JSON-like mapping."""
        if "sensor" not in data:
            raise ValueError("run config is missing required key 'sensor'")
        _reject_legacy_top_level_sections(data)
        _reject_moved_options(_section_mapping(data, "options"))

        sensor = _parse_sensor_config(data["sensor"])
        reader = ReaderConfig(**_section_mapping(data, "reader"))
        options = OptionsConfig(**_section_mapping(data, "options"))
        inversion = InversionConfig(**_section_mapping(data, "inversion"))
        clustering = ClusterConfig(**_section_mapping(data, "clustering"))
        spatial = SpatialConfig(**_section_mapping(data, "spatial"))
        canopy = CanopyConfig(**_section_mapping(data, "canopy"))

        known_keys = {
            "sensor",
            "reader",
            "options",
            "inversion",
            "clustering",
            "spatial",
            "canopy",
            "output",
            "output_policy",
            "ancillary",
            "ancillary_paths",
        }

        return cls(
            sensor=sensor,
            reader=reader,
            options=options,
            inversion=inversion,
            clustering=clustering,
            spatial=spatial,
            canopy=canopy,
            output_policy=_section_mapping(data, "output_policy", alias="output"),
            ancillary_paths=_section_mapping(data, "ancillary_paths", alias="ancillary"),
            extra={key: value for key, value in data.items() if key not in known_keys},
        )


@dataclass(frozen=True)
class SceneManifestItem:
    """Concrete paths and metadata for one scene in a batch manifest."""

    image_path: str
    background_image: str
    output_path: Optional[str] = None
    tile: Optional[str] = None
    water_year: Optional[int | str] = None
    date: Optional[str] = None
    masks: dict[str, Any] = field(default_factory=dict)
    ancillary: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        item_index: int,
    ) -> "SceneManifestItem":
        """Build a manifest item from a JSON-like mapping."""
        for key in ("image_path", "background_image"):
            if key not in data:
                raise ValueError(f"manifest scene {item_index} is missing required key {key!r}")

        known_keys = {
            "image_path",
            "background_image",
            "output_path",
            "tile",
            "water_year",
            "date",
            "masks",
            "ancillary",
        }

        return cls(
            image_path=data["image_path"],
            background_image=data["background_image"],
            output_path=data.get("output_path"),
            tile=data.get("tile"),
            water_year=data.get("water_year"),
            date=data.get("date"),
            masks=_section_mapping(data, "masks"),
            ancillary=_section_mapping(data, "ancillary"),
            extra={key: value for key, value in data.items() if key not in known_keys},
        )


@dataclass(frozen=True)
class SceneManifest:
    """JSON manifest containing concrete scene work items."""

    scenes: list[SceneManifestItem]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, manifest_file: str | Path) -> "SceneManifest":
        """Read a JSON scene manifest file."""
        data = _load_json_object(manifest_file, "scene manifest")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SceneManifest":
        """Build a scene manifest from a JSON-like mapping."""
        if "scenes" not in data:
            raise ValueError("scene manifest is missing required key 'scenes'")
        if not isinstance(data["scenes"], list):
            raise ValueError("scene manifest key 'scenes' must be a list")

        scenes = []
        for index, item in enumerate(data["scenes"]):
            if not isinstance(item, Mapping):
                raise ValueError(f"manifest scene {index} must be an object")
            scenes.append(SceneManifestItem.from_mapping(item, item_index=index))

        return cls(
            scenes=scenes,
            extra={key: value for key, value in data.items() if key != "scenes"},
        )


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"{label} loading supports JSON files only")

    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return dict(data)


def _parse_sensor_config(data: Any) -> SensorConfig:
    if isinstance(data, str):
        return SensorConfig(name=data)
    if isinstance(data, Mapping):
        return SensorConfig(**dict(data))
    raise ValueError("run config key 'sensor' must be a string or object")


def _section_mapping(
    data: Mapping[str, Any],
    key: str,
    *,
    alias: Optional[str] = None,
) -> dict[str, Any]:
    value = data.get(key)
    if value is None and alias is not None:
        value = data.get(alias)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {key!r} must be an object")
    return dict(value)


MOVED_OPTION_KEYS = {
    "target_bbox": "spatial.target_bbox",
    "target_crs": "spatial.target_crs",
    "resampling_method": "spatial.resampling_method",
    "use_custom_sensor_resolution": "spatial.use_custom_sensor_resolution",
    "clustering_method": "clustering.representative_method",
    "ignore_cloudmask_for_clustering": "clustering.ignore_cloudmask",
    "canopy_vza_adjustment": "canopy.viewable_fraction",
    "average_vertical_crown_radius": "canopy.average_vertical_crown_radius",
    "average_horizontal_crown_radius": "canopy.average_horizontal_crown_radius",
    "max_sensor_zenith": "reader.max_sensor_zenith",
    "max_solar_zenith": "reader.max_solar_zenith",
}


LEGACY_TOP_LEVEL_SECTIONS = {
    "reader_options": "reader",
    "mask_policy": "reader",
    "resampling": "spatial",
}


def _reject_moved_options(options: Mapping[str, Any]) -> None:
    moved = sorted(set(options) & set(MOVED_OPTION_KEYS))
    if moved:
        destinations = ", ".join(
            f"options.{key} -> {MOVED_OPTION_KEYS[key]}" for key in moved
        )
        raise ValueError(f"moved config option(s): {destinations}")


def _reject_legacy_top_level_sections(data: Mapping[str, Any]) -> None:
    legacy = sorted(set(data) & set(LEGACY_TOP_LEVEL_SECTIONS))
    if legacy:
        destinations = ", ".join(
            f"{key} -> {LEGACY_TOP_LEVEL_SECTIONS[key]}" for key in legacy
        )
        raise ValueError(f"moved config section(s): {destinations}")


class SpiresConfig:
    def __init__(self, config_file: str) -> None:

        self.srtmnet = None

        with open(config_file, "r") as f:
            data = json.load(f)

        _reject_legacy_top_level_sections(data)
        _reject_moved_options(_section_mapping(data, "options"))
        self.files = FilesConfig(**data["files"])
        self.sensor = SensorConfig(**data["sensor"])
        self.option = OptionsConfig(**_section_mapping(data, "options"))
        self.reader = ReaderConfig(**_section_mapping(data, "reader"))
        self.spatial = SpatialConfig(**_section_mapping(data, "spatial"))
        self.canopy = CanopyConfig(**_section_mapping(data, "canopy"))
        self.clustering = ClusterConfig(**_section_mapping(data, "clustering"))
        self.lut = LookUpTableConfig(**_section_mapping(data, "lut"))
        self.inv = InversionConfig(**_section_mapping(data, "inversion"))

        if self.canopy.viewable_fraction:
            if self.files.canopy_fraction is None:
                raise ValueError(
                    "If setting canopy.viewable_fraction to true, "
                    "files.canopy_fraction must be provided."
                )

        if self.option.ignore_topography_correction:
            self.sensor.apply_topo_correction = False

        if self.sensor.apply_topo_correction:
            print("WARNING: topo correction not yet implemented.")
        # TODO topo correction not yet implemented
        # if self.sensor.apply_topo_correction:
        #    for f in ["dem", "slope", "aspect"]:
        #        path = Path(getattr(self.files, f) or "")
        #        if not path.is_file():
        #            raise FileNotFoundError(f"Topography data {f} missing at {path}")
        #
        #    pkg = importlib.resources.files("spires_io.data.topo_correction_lut")
        #    self.srtmnet = pkg.joinpath("srtmnet_coarse.nc")
