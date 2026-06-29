import numpy as np
import importlib.resources
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

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
    cpu_cores: int = 1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_bbox: Optional[list] = None
    target_crs: Optional[str] = None
    resampling_method: str = "bilinear"
    use_custom_sensor_resolution: bool = False
    clustering_method: Optional[str] = None
    ignore_cloudmask_for_clustering: bool = False
    canopy_vza_adjustment: bool = False
    ignore_topography_correction: bool = True
    average_vertical_crown_radius: float = 4.644
    average_horizontal_crown_radius: float = 1.72
    atmosphere_aod: float = 0.10
    atmosphere_watervapor_gcm2: float = 0.5
    max_sensor_zenith: float = 65.0
    max_solar_zenith: float = 85.0

    def __post_init__(self) -> None:
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be >= 1")

        if self.resampling_method is not None:
            if self.resampling_method.lower() not in [
                "average",
                "bilinear",
                "cubic",
                "med",
                "nearest",
            ]:
                raise ValueError(f"Invalid resampling method: {self.resampling_method}")

        self.atmosphere_aod = max(
            constants.MIN_AOD, min(self.atmosphere_aod, constants.MAX_AOD)
        )

        self.atmosphere_watervapor_gcm2 = max(
            constants.MIN_H2O, min(self.atmosphere_watervapor_gcm2, constants.MAX_H2O)
        )

        self.max_sensor_zenith = max(
            constants.MIN_ZENITH, min(self.max_sensor_zenith, constants.MAX_ZENITH)
        )

        self.max_solar_zenith = max(
            constants.MIN_ZENITH, min(self.max_solar_zenith, constants.MAX_ZENITH)
        )


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


class SpiresConfig:
    def __init__(self, config_file: str) -> None:

        self.srtmnet = None

        with open(config_file, "r") as f:
            data = json.load(f)

        self.files = FilesConfig(**data["files"])
        self.sensor = SensorConfig(**data["sensor"])
        self.option = OptionsConfig(**data.get("options", {}))
        self.lut = LookUpTableConfig(**data.get("lut", {}))
        self.inv = InversionConfig(**data.get("inversion", {}))

        if self.option.canopy_vza_adjustment:
            if self.files.canopy_fraction is None:
                raise ValueError(
                    f"If setting canopy vza adjustment to be true a canopy fraction data must be provided."
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
