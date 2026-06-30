"""VIIRS surface-reflectance grid and data-field names."""

VIIRS_1KM_GRID = "HDFEOS/GRIDS/VIIRS_Grid_1km_2D"
VIIRS_500M_GRID = "HDFEOS/GRIDS/VIIRS_Grid_500m_2D"
VIIRS_1KM_GRID_NAME = "VIIRS_Grid_1km_2D"
VIIRS_500M_GRID_NAME = "VIIRS_Grid_500m_2D"

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
