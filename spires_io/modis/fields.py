"""MODIS surface-reflectance grid and data-field names."""

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
