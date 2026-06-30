"""Producer-side checks for the spires-io -> spires-inversion boundary."""

from spires_contract.spectra import validate_solar_angles, validate_target_spectra

from spires_io.modis import prepare_modis_scene_for_inversion
from spires_io.viirs import prepare_viirs_scene_for_inversion
from test_modis_load_surface_reflectance import build_mock_modis_raw_dataset
from test_viirs_load_surface_reflectance import build_mock_viirs_raw_dataset


def _validate_scene_arrays(scene):
    target_spectra = scene["reflectance"].astype("float64")
    solar_angles = scene["solar_zenith"].astype("float64")

    validate_target_spectra(target_spectra)
    validate_solar_angles(solar_angles)


def test_viirs_prepared_scene_emits_contract_valid_target_and_solar_arrays():
    scene = prepare_viirs_scene_for_inversion(build_mock_viirs_raw_dataset())

    _validate_scene_arrays(scene)


def test_modis_prepared_scene_emits_contract_valid_target_and_solar_arrays():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())

    _validate_scene_arrays(scene)
