import re


# List of supported sensors
SUPPORTED_SENSORS = ["viirs", "modis", "sentinel-2", "emit"]

DEFAULT_RESOLUTION = {"viirs": 500.0, 
                      "modis": 500.0, 
                      "sentinel-2": 10.0, 
                      "emit": 60.0}




# TODO do we need these?
VIIRS_FILENAME_RE = re.compile(
    r"^(?P<product>VNP09GA|VJ109GA|VJ209GA)\.A(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?P<tile>h\d{2}v\d{2})\.(?P<collection>\d{3})\.(?P<processing>\d+)\.h5$"
)

MODIS_FILENAME_RE = re.compile(
    r"^(?P<product>MOD09GA|MYD09GA)\.A(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?P<tile>h\d{2}v\d{2})\.(?P<collection>\d{3})\.(?P<processing>\d+)\.(?P<suffix>hdf|h5)$",
    re.IGNORECASE,
)




# MODIS constants
MODIS_DEFAULT_BAND_NAMES = ("1", "2", "3", "4", "5", "6", "7")
MODIS_ANALYSIS_BANDS = MODIS_DEFAULT_BAND_NAMES

MODIS_1KM_GRID_NAME = "MODIS_Grid_1km_2D"
MODIS_500M_GRID_NAME = "MODIS_Grid_500m_2D"
MODIS_500M_REFLECTANCE_BANDS = ("1", "2", "3", "4", "5", "6", "7")
MODIS_1KM_GEOMETRY_FIELDS = {
    "sensor_zenith": "SensorZenith_1",
    "sensor_azimuth": "SensorAzimuth_1",
    "solar_zenith": "SolarZenith_1",
    "solar_azimuth": "SolarAzimuth_1",
}
MODIS_1KM_QA_FIELDS = {
    "state_1km": "state_1km_1",
    "num_observations_1km": "num_observations_1km",
    "range_1km": "Range_1",
    "gflags_1km": "gflags_1",
    "orbit_pnt_1km": "orbit_pnt_1",
    "granule_pnt_1km": "granule_pnt_1",
}
MODIS_500M_SUPPORT_FIELDS = {
    "num_observations_500m": "num_observations_500m",
    "qc_500m": "QC_500m_1",
    "obscov_500m": "obscov_500m_1",
    "iobs_res_500m": "iobs_res_1",
    "q_scan_500m": "q_scan_1",
}






# VIIRS constants
VIIRS_1KM_REFLECTANCE_BANDS = (
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M7",
    "M8",
    "M10",
    "M11",
)

VIIRS_500M_REFLECTANCE_BANDS = (
    "I1",
    "I2",
    "I3",
)

VIIRS_1KM_GEOMETRY_FIELDS = {
    "solar_zenith": "SolarZenith_1",
    "solar_azimuth": "SolarAzimuth_1",
    "sensor_zenith": "SensorZenith_1",
    "sensor_azimuth": "SensorAzimuth_1",
}

VIIRS_1KM_QA_FIELDS = {
    "qa_qf1": "SurfReflect_QF1_1",
    "qa_qf2": "SurfReflect_QF2_1",
    "qa_qf3": "SurfReflect_QF3_1",
    "qa_qf4": "SurfReflect_QF4_1",
    "qa_qf5": "SurfReflect_QF5_1",
    "qa_qf6": "SurfReflect_QF6_1",
    "qa_qf7": "SurfReflect_QF7_1",
    "land_water_mask": "land_water_mask_1",
    "num_observations_1km": "num_observations_1km",
    "obscov_1km": "obscov_1km_1",
    "orbit_pnt": "orbit_pnt_1",
}

VIIRS_500M_SUPPORT_FIELDS = {
    "iobs_res": "iobs_res_1",
    "num_observations_500m": "num_observations_500m",
    "obscov_500m": "obscov_500m_1",
}

VIIRS_ANALYSIS_BANDS = VIIRS_500M_REFLECTANCE_BANDS + VIIRS_1KM_REFLECTANCE_BANDS






# Shared constants
VALID_EXTENSIONS = [".tif", ".nc", ".h5", ".mat"]

VIIRS_MODIS_CRS = (
    'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",'
    'DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Sinusoidal"],'
    'PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["Meter",1]]'
)


# For topographic correction
MIN_MU_S = 0.25
MIN_AOD = 0.03
MAX_AOD = 1.0
MIN_H2O = 0.2
MAX_H2O = 5.0

# STATIC DATA
STATIC_DATA = ["canopy_fraction", "dem", "slope", "aspect", "skyview"] 