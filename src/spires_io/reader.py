import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
import rasterio as rio
from joblib import Parallel, delayed
import netCDF4 as nc
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import zoom, map_coordinates
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

import spires_io.constants as constants
from spires_io.configs import SpiresConfig


class SpiresData:
    def __init__(self, config_file:str):

        # Validate input config
        self.config = SpiresConfig(config_file=config_file)

        self.sensor_name = self.config.sensor.name.lower()
        self.sensor_resolution = self.config.sensor.resolution

        self.dem = None
        self.slope = None
        self.skyview = None
        self.aspect = None
        self.canopy_fraction = None
        self.sensor_zenith = None
        self.sensor_azimuth = None
        self.solar_zenith = None
        self.solar_azimuth = None
        self.target_spectra = None
        self.background_spectra = None

        self.lut_dir = None
        self.lut_dif = None

        sensor_loaders = {
            "modis": self._load_modis,
            "viirs": self._load_viirs,
            "sentinel-2": self._load_s2,
            "emit": self._load_emit,
        }
        self.load_sensor_data = sensor_loaders.get(self.sensor_name)


    # The two main methods that user has access to Load everything specificed in the config
    # and after which, can run cluster if this is somethign they want to do
    def load(self) -> None:
        
        # Load all static data first
        for f in constants.STATIC_DATA:
            with rio.open(getattr(self.config.files, f)) as src:
                data = src.read().squeeze()
                if self.config.option.resampling_method is not None:
                    if f != "aspect":
                        data = self._warp_data(data, src.transform, src.crs)
                    else:
                        sin_aspect = np.sin(np.radians(data))
                        cos_aspect = np.cos(np.radians(data))
                        aspect_data = np.stack([sin_aspect, cos_aspect], axis=0)
                        aspect_data_warped = self._warp_data(
                            aspect_data, src.transform, src.crs
                        )
                        data = (
                            np.degrees(
                                np.arctan2(
                                    aspect_data_warped[..., 0], aspect_data_warped[..., 1]
                                )
                            )
                            % 360
                        )
                        data = data[..., np.newaxis]

                if f == "canopy_fraction":
                    if np.nanmax(data) > 1.0:
                        data = data / 100.0
                    data[np.isnan(data)] = 0.0
                    data[data>100.0] = np.nan

                if f == "slope":
                    data[data<0.0] = np.nan
                    data[data>90.0] = np.nan
                if f == "aspect":
                    data[data<0.0] = np.nan
                    data[data>360.0] = np.nan
                if f == "dem":
                    data[data>8000.0] = np.nan
                    data[data<-1000.0] = np.nan
                if f == "skyview":
                    data[data>1.0] = np.nan
                    data[data<0.0] = np.nan


                
        # Now the image data
        img_path = Path(self.config.files.image_data)

        if img_path.is_dir():
            all_img_files = [
                f
                for f in img_path.iterdir()
                if f.is_file()
                and not f.name.startswith(".")
                and (f.suffix.lower() in constants.VALID_EXTENSIONS or f.suffix == "")
            ]
        else:
            all_img_files = [img_path]
        
        img_files = self._filter_by_date(all_img_files)

        if not img_files:
            raise FileNotFoundError(
                f"No files matching configuration found at {img_path}"
            )
        
        if self.config.option.cpu_cores > 1 and len(img_files) > 1:
            data = Parallel(n_jobs=self.config.option.cpu_cores)(
                delayed(self.load_sensor_data)(f) for f in img_files
            )
        else:
            data = [self.load_sensor_data(f) for f in img_files]
        
        self.target_spectra = np.stack([r[0] for r in data], axis=-1)
        self.sensor_zenith = np.stack([r[1][..., 0] for r in data], axis=-1)
        self.sensor_azimuth = np.stack([r[1][..., 1] for r in data], axis=-1)
        self.solar_zenith = np.stack([r[1][..., 2] for r in data], axis=-1)
        self.solar_azimuth = np.stack([r[1][..., 3] for r in data], axis=-1)

        print(self.target_spectra.shape)
        print(self.sensor_zenith.shape)
        print(self.sensor_azimuth.shape)
        print(self.solar_zenith.shape)

        # Next, load r0 - background spectra
        self.background_spectra = self._load_r0()


        # Then, we load any sort of masks that were passed
        # TODO to look at, was water mask in VIIRS and MODIS layers? how to handle if so, with different sensors
        #

        

        print(self.target_spectra.shape)
        print(self.sensor_zenith.shape)
        print(self.sensor_azimuth.shape)
        print(self.solar_zenith.shape)
        import matplotlib.pyplot as plt
        plt.imshow(self.background_spectra[:,:,0])
        plt.show()


        self._validate_dimensions()



    def cluster(self, method) -> None:   
        pass

    

    # Everything below here will be incorperated into the load or cluster
    # so users can run spires_data.load() .... spires_data.cluster()


    #####
    #####
    #####


    def _validate_dimensions(self) -> None:

        ref_shape = self.target_spectra.shape[:2]
        
        attrs_to_check = {
            "dem": self.dem,
            "slope": self.slope,
            "skyview": self.skyview,
            "aspect": self.aspect,
            "canopy_fraction": self.canopy_fraction,
            "sensor_zenith": self.sensor_zenith,
            "sensor_azimuth": self.sensor_azimuth,
            "solar_zenith": self.solar_zenith,
            "solar_azimuth": self.solar_azimuth,
            "background_spectra": self.background_spectra
        }
        
        for name, data in attrs_to_check.items():
            if data is not None:
                if data.shape[:2] != ref_shape:
                    raise ValueError(
                        f"Dimension mismatch in '{name}': "
                        f"Expected {ref_shape}, got {data.shape[:2]}"
                    )
                

    def _load_r0(self) -> npt.NDArray[np.float32]:

        r0_path = Path(self.config.files.snowfree_image)

        # Handle if the user just gives an r0 that is a sensor file
        # which is valid, especially if there is limited data
        if r0_path.suffix.lower() in [".h5", ".nc"]:
            r0 , _ = self.load_sensor_data(r0_path)
        else:
            # Otherwise, we will assume the r0 is a something we can open with rasterio
            with rio.open(self.config.files.snowfree_image) as src:
                r0 = src.read()
                if self.config.option.resampling_method is not None:
                    r0 = self._warp_data(r0, src.transform, src.crs)

        if r0.shape[-1] != self.target_spectra.shape[2]:
             raise ValueError(f"Snow free image has different number of bands (len={len(r0.shape[-1])}) compared to input wavelengths (len={len(self.target_spectra.shape[2])}))")
        
        return r0.astype(np.float32)



    def _load_modis(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:


        # ? bands = self.config.sensor.selected_bands or (
        #    list(constants.VIIRS_500M_REFLECTANCE_BANDS) + 
        #    list(constants.VIIRS_1KM_REFLECTANCE_BANDS)
        #)



        # Determine if we need to apply correction based on sensor/version
        #if not self.config.option.ignore_topography_correction:
        #    self._determine_topo_correction()

        pass









    def _load_viirs(self, img_file) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

        ds = nc.Dataset(img_file, "r")

        bands = self.config.sensor.selected_bands or (
            list(constants.VIIRS_500M_REFLECTANCE_BANDS) + 
            list(constants.VIIRS_1KM_REFLECTANCE_BANDS)
        )

        def process_band(band_id: str):
            is_500m = band_id in constants.VIIRS_500M_REFLECTANCE_BANDS
            grid_name = "VIIRS_Grid_500m_2D" if is_500m else "VIIRS_Grid_1km_2D"
            field_name = f"SurfReflect_{band_id}_1"
            data = ds["HDFEOS"]["GRIDS"][grid_name]["Data Fields"][field_name][:].astype(np.float32)
            return data if is_500m else zoom(data, 2, order=3)

        data = np.stack([process_band(b) for b in bands], axis=0)

        geom_fields = [
            constants.VIIRS_1KM_GEOMETRY_FIELDS["sensor_zenith"],
            constants.VIIRS_1KM_GEOMETRY_FIELDS["sensor_azimuth"],
            constants.VIIRS_1KM_GEOMETRY_FIELDS["solar_zenith"],
            constants.VIIRS_1KM_GEOMETRY_FIELDS["solar_azimuth"],
        ]

        geom = np.stack(
            [
                zoom(
                    ds["HDFEOS"]["GRIDS"]["VIIRS_Grid_1km_2D"]["Data Fields"][f][
                        :
                    ].astype(np.float32),
                    2,
                    order=1,
                )
                for f in geom_fields
            ],
            axis=0,
        )

        if self.config.option.resampling_method is not None:

            west = ds.getncattr("WestBoundingCoord")
            north = ds.getncattr("NorthBoundingCoord")
            west, north = Transformer.from_crs(
                "EPSG:4326", constants.VIIRS_MODIS_CRS, always_xy=True
            ).transform(west, north)

            transform = Affine(
                self.sensor_resolution, 0, west, 0, -self.sensor_resolution, north
            )

            data = self._warp_data(data, transform, constants.VIIRS_MODIS_CRS)

            # VZA, VAA, SZA, SAA
            # geom[..., 0], geom[..., 1], geom[..., 2], geom[..., 3]
            geom = self._warp_data(geom, transform, constants.VIIRS_MODIS_CRS)
        
        # Determine if we need to apply correction based on sensor/version
        #if not self.config.option.ignore_topography_correction:
        #    warped_target = self._determine_topo_correction(warped_target)

        
        return data, geom








    def _load_s2(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

        # ? bands = self.config.sensor.selected_bands or (
        #    list(constants.VIIRS_500M_REFLECTANCE_BANDS) + 
        #    list(constants.VIIRS_1KM_REFLECTANCE_BANDS)
        #)

        pass
    





    def _load_emit(self, img_file: Path) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

        # ? bands = self.config.sensor.selected_bands or (
        #    list(constants.VIIRS_500M_REFLECTANCE_BANDS) + 
        #    list(constants.VIIRS_1KM_REFLECTANCE_BANDS)
        #)

        with nc.Dataset(img_file) as ds:
            glt_x = ds.groups['location']['glt_x'][:].flatten().astype(int) - 1
            glt_y = ds.groups['location']['glt_y'][:].flatten().astype(int) - 1
            rows, cols = ds.groups['location']['glt_x'].shape

        with xr.open_dataset(img_file) as ds:
            downtrack, crosstrack, bands = ds.reflectance.shape
            raw_data = ds["reflectance"].values.reshape(-1, bands).astype(np.float32)

        valid_mask = (glt_x >= 0) & (glt_y >= 0)
        valid_indices = glt_y[valid_mask] * crosstrack + glt_x[valid_mask]
        
        flat_data = np.full((bands, rows * cols), np.nan, dtype=np.float32)
        flat_data[:, valid_mask] = raw_data[valid_indices].T
        data = flat_data.reshape(bands, rows, cols)
     
        # NOTE Assumes near-nadir for ISS EMIT observation
        _, rows, cols = data.shape
        vza = np.zeros((rows, cols), dtype=np.float32)
        vaa = np.zeros((rows, cols), dtype=np.float32)

        # TODO SZA SAA from pysolar because not in reflectance data product
        sza = np.zeros((rows, cols), dtype=np.float32)
        saa = np.zeros((rows, cols), dtype=np.float32)     

        geom = np.stack([vza, vaa, sza, saa])

        # Determine if we need to apply correction based on sensor/version
        #if not self.config.option.ignore_topography_correction:
        #    warped_target = self._determine_topo_correction(warped_target)
        
        # Apply cleaning of some noisy wavelengths around deep water features
        mask = (self.config.sensor.wavelength < 495) | \
               ((self.config.sensor.wavelength >= 1325) & (self.config.sensor.wavelength < 1468)) | \
               ((self.config.sensor.wavelength >= 1765) & (self.config.sensor.wavelength <= 1967))
        data[mask, :, :] = np.nan


        if self.config.option.resampling_method is not None:
            src_transform = Affine(ds.geotransform[1],  ds.geotransform[2],  
                                ds.geotransform[0],  ds.geotransform[4],  
                                ds.geotransform[5], ds.geotransform[3])
            
            data = self._warp_data(data.reshape(bands, rows, cols), 
                                            src_transform, 
                                            'EPSG:4326')
            
            geom = self._warp_data(geom, src_transform, 'EPSG:4326')


        return data, geom











    def _go_viewable_gap_fraction_adjustment(self) -> None:

        # See Liu et al 2008 for example. Default is b_R=2.7 to represent Lodgepole Pine (Bair et al., 2021).
        b_R = self.config.option.average_vertical_crown_radius / self.config.option.average_horizontal_crown_radius
        
        theta_v_prime = np.arctan(b_R * np.tan(np.radians(self.vza)))
        theta_s_prime = np.radians(
            90 - np.degrees(np.arctan((b_R * np.tan(np.radians(90 - self.slope)))))
        )
        phi_v_prime = np.radians(self.vaa - self.aspect)
        self.canopy_fraction = 1 - (
            (1 - self.canopy_fraction)
            ** (
                (np.cos(theta_s_prime))
                / (
                    np.cos(phi_v_prime) * np.sin(theta_v_prime) * np.sin(theta_s_prime)
                    + np.cos(theta_v_prime) * np.cos(theta_s_prime)
                )
            )
        )








    def _perform_topo_correction(self, image) -> npt.NDArray[np.float32]:

        # Cache in case we are running for many scenes
        if self.lut_dir is None or self.lut_dif is None:
            ds = xr.open_dataset(self.files.atm_lut)
            
            dims = ("AOT550", "H2OSTR", "surface_elevation_km", "solar_zenith", "wl")
            points = (ds.AOT550.values, ds.H2OSTR.values, ds.surface_elevation_km.values, ds.solar_zenith.values)

            data_dir = ds.transm_down_dir.transpose(*dims).values
            data_dif = ds.transm_down_dif.transpose(*dims).values
            
            self.lut_dir = RegularGridInterpolator(points, data_dir, bounds_error=False, fill_value=None)
            self.lut_dif = RegularGridInterpolator(points, data_dif, bounds_error=False, fill_value=None)
            
            self.lut_wl = ds.wl.values
            ds.close()

        # TODO For now just testing for completion
        # there is a fixed SZA etc
        rows, cols = image.shape[:2]
        aod_arr = np.full((rows, cols), self.config.option.atmosphere_aod)
        wv_arr = np.full((rows, cols), self.config.option.atmosphere_watervapor_gcm2)
        alt_arr = self.dem / 1e3 
        sza_arr = np.full((rows, cols), self.sza)
        img_grid = np.stack([aod_arr, wv_arr, alt_arr, sza_arr], axis=-1)
        
        Edir = self.lut_dir(img_grid)
        Edif = self.lut_dif(img_grid)

        # Interpolate coarse RT sims to sensor wavelengths
        Edir = interp1d(self.lut_wl, Edir, axis=-1, kind='linear', fill_value="extrapolate")(self.wavelength)
        Edif = interp1d(self.lut_wl, Edif, axis=-1, kind='linear', fill_value="extrapolate")(self.wavelength)

        # Assume hooking effect after 1500 nm is negligible (Bair et al., 2025)
        # In other words, this assumes 100% direct light after 1500 nm
        atm_lut_mask = self.wavelength > 1500.0
        Edir[..., atm_lut_mask] = 1.0
        Edif[..., atm_lut_mask] = 0.0

        mu_0 = np.cos(np.radians(self.sza))
        mu_s = (np.cos(np.radians(self.sza)) * np.cos(np.radians(self.slope)) + 
                np.sin(np.radians(self.sza)) * np.sin(np.radians(self.slope)) * np.cos(np.radians(self.saa - self.aspect)))

        # apply terrain shadowing
        shadow_mask = self._apply_shadow_mask()
        mu_s = shadow_mask * mu_s

        # limit the correction for shaded / low signal slopes
        mu_s = np.clip(mu_s, constants.MIN_MU_S, 1.0)

        # apply correction for terrain hooking
        denom = (Edir * (mu_s / mu_0) + Edif * self.skyview)
        denom = np.clip(denom, 1e-6, None)
        topo_corrected_image = image * ((Edir + Edif) / denom)

        # Post hoc correction to scale really bright data
        idx_550 = np.argmin(np.abs(self.wavelength - 550.0))
        scale_factor = topo_corrected_image[..., idx_550]
        scale_mask = scale_factor > 1.10
        topo_corrected_image[scale_mask, :] /= scale_factor[scale_mask, np.newaxis]

        return topo_corrected_image.astype(np.float32)










    def _apply_shadow_mask(self) -> npt.NDArray[np.float32]:
        """TODO can store this too in memory for user
        """
        rows, cols = self.dem.shape
        pixel_list = [(i, j) for i in range(rows) for j in range(cols)]
        results = Parallel(n_jobs=self.config.option.cpu_cores)(
            delayed(self._ray_trace_pixel)(i, j) for i, j in pixel_list
        )
        shadow_mask = np.array(results).reshape(rows, cols)
        return shadow_mask[..., np.newaxis]
    

    def _ray_trace_pixel(self, i, j):

        pix_size = self.sensor_resolution
        i_lim, j_lim = self.dem.shape
        
        # ray path
        tan_theta_e = np.tan(np.radians(90 - self.sza))
        tan_sundir = -1 * np.tan(np.radians(self.saa))
        
        # Search length # TODO to be constant
        PIXEL_SEARCH_LENGTH_RAY_TRACE = 300
        y_mover = np.arange(0, PIXEL_SEARCH_LENGTH_RAY_TRACE+0.1, 1)
        if self.saa > 270 or self.saa < 90: y_mover *= -1
        x_mover = np.round(y_mover * tan_sundir).astype(int)
        
        y, x = i + y_mover, j + x_mover
        
        # filter out of bounds
        mask = (y < i_lim-1) & (x < j_lim-1) & (y >= 0) & (x >= 0)
        y, x = y[mask], x[mask]
        
        # interpolate
        zi = map_coordinates(self.dem, np.vstack((y, x)), order=1)
        h = (np.sqrt(((y-i)*pix_size)**2 + ((x-j)*pix_size)**2)) * tan_theta_e + self.dem[i,j]
        
        # 0 = shadow ; 1=no shadow
        return 0 if ((h[1:] < zi[1:]).any()) else 1







    def _warp_data(self, source, src_transform, src_crs) -> npt.NDArray[np.float32]:

        src_crs_obj = CRS.from_string(src_crs) if isinstance(src_crs, str) else src_crs
        dst_crs_obj = CRS.from_string(self.config.option.target_crs)

        left, bottom, right, top = self.config.option.target_bbox
        method = getattr(Resampling, self.config.option.resampling_method.lower())

        x_res = self.sensor_resolution
        y_res = self.sensor_resolution
        width = int((right - left) / x_res)
        height = int((top - bottom) / y_res)

        dst_transform = rio.transform.from_origin(left, top, x_res, y_res)

        if source.ndim == 2:
            source = source[np.newaxis, ...]
        source = source.astype(np.float32)

        bands, _, _ = source.shape
        dst_array = np.zeros((bands, height, width), dtype=np.float32)

        for i in range(bands):
            src_band = np.ascontiguousarray(source[i, :, :])
            dst_band = np.zeros((height, width), dtype=np.float32)

            reproject(
                source=src_band,
                destination=dst_band,
                src_transform=src_transform,
                src_crs=src_crs_obj,
                dst_transform=dst_transform,
                dst_crs=dst_crs_obj,
                resampling=method,
            )
            dst_array[i, :, :] = dst_band

        return dst_array.transpose(1, 2, 0)


    def _filter_by_date(self, file_list: List[Path]) -> List[Path]:
        if not (self.config.option.start_date or self.config.option.end_date):
            return sorted(file_list)
            
        start = datetime.strptime(self.config.option.start_date, "%Y-%m-%d") if self.config.option.start_date else datetime.min
        end = datetime.strptime(self.config.option.end_date, "%Y-%m-%d") if self.config.option.end_date else datetime.max
        
        filtered = []
        for f in file_list:
            f_date = self._get_date_from_str(f.name)
            if f_date and (start <= f_date <= end):
                filtered.append(f)
        return sorted(filtered)




    def _get_date_from_str(self, filename: str) -> Optional[datetime]:
        parts = re.findall(r'\d+', filename)
        for part in parts:
            for fmt in ("%Y%j", "%Y%m%d"):
                try:
                    return datetime.strptime(part, fmt)
                except ValueError:
                    continue
        return None



# TESTING
data = SpiresData("/Users/bawilder/Code/SPIReS/spires-io/example_config.json")

data.load()

