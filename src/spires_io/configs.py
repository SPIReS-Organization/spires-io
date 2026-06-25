import numpy as np
import importlib.resources
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import spires_io.constants as constants


@dataclass
class FilesConfig:
    image_data: str
    snowfree_image: str
    lut: str
    cloud_mask: Optional[str] = None
    water_mask: Optional[str] = None
    dem: Optional[str] = None
    slope: Optional[str] = None
    aspect: Optional[str] = None
    skyview: Optional[str] = None
    canopy_fraction: Optional[str] = None

    def __post_init__(self) -> None:
        for f in ["image_data", "snowfree_image", "lut"]:
            path = getattr(self, f)
            if path is None:
                raise ValueError(
                    f"The following data are required: ['image_data', 'snowfree_image','lut']"
                )

            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"The following data ({f}) were not found at: {path}"
                )
            if path.is_file() and path.suffix.lower() not in constants.VALID_EXTENSIONS:
                raise ValueError(f"{path} has invalid file type: {path.suffix}")


@dataclass
class SensorConfig:
    name: str
    product_version: Optional[str] = None
    selected_bands: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.name.lower() not in constants.SUPPORTED_SENSORS:
            raise ValueError(
                f"Sensor name '{self.name}' not in supported: {constants.SUPPORTED_SENSORS}"
            )

        self.resolution = constants.DEFAULT_RESOLUTION.get(self.name.lower())
        pkg = importlib.resources.files("spires_io.data.wavelengths")
        self.wavelength_file = pkg.joinpath(f"{self.name.lower()}.txt")

        if not self.wavelength_file.is_file():
            raise FileNotFoundError(f"Could not find wavelength file for {self.name}")

        wl_data = np.genfromtxt(
            self.wavelength_file,
            dtype=None,
            names=["band", "wl", "fwhm"],
            encoding="utf-8",
            usecols=(0, 1, 2),
        )
        self.band_names, self.wavelength, self.fwhm = (
            wl_data["band"],
            wl_data["wl"],
            wl_data["fwhm"],
        )

        if self.selected_bands:
            mask = np.isin(self.band_names, self.selected_bands)
            self.band_names, self.wavelength, self.fwhm = (
                self.band_names[mask],
                self.wavelength[mask],
                self.fwhm[mask],
            )

        self.apply_topo_correction = False
        if "modis" in self.name.lower():
            self.apply_topo_correction = True
        if "viirs" in self.name.lower():
            self.apply_topo_correction = True
        if "sentinel" in self.name.lower():
            self.apply_topo_correction = False
        if (
            "emit" in self.name.lower()
            and self.product_version
            and "1" in self.product_version
        ):
            self.apply_topo_correction = True


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
    canopy_vza_adjustment: bool = True
    ignore_topography_correction: bool = False
    average_vertical_crown_radius: float = 4.644
    average_horizontal_crown_radius: float = 1.72
    atmosphere_aod: float = 0.10
    atmosphere_watervapor_gcm2: float = 0.5

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

        self.atmosphere_aod = max(constants.MIN_AOD, min(self.atmosphere_aod, constants.MAX_AOD))

        self.atmosphere_watervapor_gcm2 = max(
            constants.MIN_H2O, min(self.atmosphere_watervapor_gcm2, constants.MAX_H2O)
        )


@dataclass
class LookUpTableConfig:
    reflectance: str = "reflectance"
    grain_radius: str = "grain_radius"
    pollutant: str = "dust"
    liquid_water_fraction: Optional[float] = None
    solar_zenith: str = "solar_zenith"


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

        with open(config_file, "r") as f:
            data = json.load(f)

        self.files = FilesConfig(**data["files"])
        self.sensor = SensorConfig(**data["sensor"])
        self.option = OptionsConfig(**data.get("options", {}))
        self.lut = LookUpTableConfig(**data.get("lut", {}))
        self.inv = InversionConfig(**data.get("inversion", {}))

        if self.option.ignore_topography_correction:
            self.sensor.apply_topo_correction = False

        if self.sensor.apply_topo_correction:
            for f in ["dem", "slope", "aspect"]:
                path = Path(getattr(self.files, f) or "")
                if not path.is_file():
                    raise FileNotFoundError(f"Topography data {f} missing at {path}")



