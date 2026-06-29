import numpy as np
import pytest
import xarray as xr

from spires_io import SpiresData
from spires_io.modis import prepare_modis_scene_for_inversion
from spires_io.viirs import prepare_viirs_scene_for_inversion
from test_modis_hdf import build_mock_modis_raw_dataset
from test_viirs_hdf import build_mock_viirs_raw_dataset


def _background_for(scene):
    return xr.DataArray(
        np.ones(scene["reflectance"].shape, dtype=np.float32),
        dims=("y", "x", "band"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x", "band")},
    )


def _mask_for(scene, values):
    return xr.DataArray(
        np.array(values, dtype=bool),
        dims=("y", "x"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x")},
    )


def _without_spatial_ref(data_array):
    return data_array.drop_vars("spatial_ref", errors="ignore")


def test_from_scene_wraps_modis_prepared_scene():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())

    data = SpiresData.from_scene(scene)

    assert data.target_spectra.identical(scene["reflectance"])
    assert data.solar_zenith.identical(scene["solar_zenith"])
    assert data.valid_mask.identical(scene["valid_inversion_mask"])
    assert data.background_spectra is None


def test_from_scene_wraps_viirs_prepared_scene():
    scene = prepare_viirs_scene_for_inversion(build_mock_viirs_raw_dataset())

    data = SpiresData.from_scene(scene)

    assert data.target_spectra.dims == ("y", "x", "band")
    assert data.solar_zenith.dims == ("y", "x")
    assert data.valid_mask.dims == ("y", "x")


def test_from_scene_rejects_missing_required_variables():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    scene = scene.drop_vars("valid_inversion_mask")

    with pytest.raises(ValueError, match="missing required variables"):
        SpiresData.from_scene(scene)


def test_from_scene_rejects_noncanonical_reflectance_dims():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    scene["reflectance"] = scene["reflectance"].transpose("band", "y", "x")

    with pytest.raises(ValueError, match="reflectance.*dims"):
        SpiresData.from_scene(scene)


def test_assign_background_returns_new_object_and_leaves_original_unchanged():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    data = SpiresData.from_scene(scene)
    background = _background_for(scene)

    updated = data.assign_background(background)

    assert updated is not data
    assert data.background_spectra is None
    assert updated.background_spectra.identical(background)
    assert "background_reflectance" not in data.to_dataset()

    expected_background = background.copy()
    expected_background.name = "background_reflectance"
    actual = _without_spatial_ref(updated.to_dataset()["background_reflectance"])
    assert actual.identical(expected_background)


def test_assign_background_rejects_coordinate_mismatch():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    background = _background_for(scene).assign_coords(x=[10, 11])

    with pytest.raises(ValueError, match="coordinate 'x' does not match"):
        SpiresData.from_scene(scene).assign_background(background)


def test_assign_mask_stores_component_and_updates_valid_mask():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    data = SpiresData.from_scene(scene)
    mask = _mask_for(scene, [[True, False], [False, True]])

    updated = data.assign_mask("manual", mask)

    assert "mask_manual" not in data.scene

    expected_mask = mask.copy()
    expected_mask.name = "mask_manual"
    actual_mask = _without_spatial_ref(updated.scene["mask_manual"])
    xr.testing.assert_equal(actual_mask, expected_mask)

    expected_external = mask.copy()
    expected_external.name = "mask_external_inversion"
    actual_external = _without_spatial_ref(updated.scene["mask_external_inversion"])
    xr.testing.assert_equal(actual_external, expected_external)

    expected_valid = scene["valid_inversion_mask"] & (~mask)
    expected_valid.name = "valid_inversion_mask"
    xr.testing.assert_identical(updated.valid_mask, expected_valid)


def test_assign_masks_ors_external_masks():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    mask_a = _mask_for(scene, [[True, False], [False, False]])
    mask_b = _mask_for(scene, [[False, False], [False, True]])

    updated = SpiresData.from_scene(scene).assign_masks({"a": mask_a, "b": mask_b})

    expected_external = mask_a | mask_b
    expected_external.name = "mask_external_inversion"
    actual_external = _without_spatial_ref(updated.scene["mask_external_inversion"])
    xr.testing.assert_equal(actual_external, expected_external)

    expected_valid = scene["valid_inversion_mask"] & (~expected_external)
    expected_valid.name = "valid_inversion_mask"
    xr.testing.assert_identical(updated.valid_mask, expected_valid)


def test_assign_mask_rejects_non_yx_mask():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    mask = xr.DataArray(np.ones((2, 2), dtype=bool), dims=("row", "col"))

    with pytest.raises(ValueError, match="mask must have dims"):
        SpiresData.from_scene(scene).assign_mask("bad", mask)


def test_assign_mask_rejects_coordinate_mismatch():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    mask = _mask_for(scene, [[True, False], [False, True]]).assign_coords(y=[10, 11])

    with pytest.raises(ValueError, match="coordinate 'y' does not match"):
        SpiresData.from_scene(scene).assign_mask("bad", mask)


def test_inversion_inputs_requires_background():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())

    with pytest.raises(ValueError, match="background must be assigned"):
        SpiresData.from_scene(scene).inversion_inputs()


def test_inversion_inputs_returns_float64_arrays_and_updated_valid_mask():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())
    background = _background_for(scene)
    mask = _mask_for(scene, [[False, True], [False, False]])

    data = SpiresData.from_scene(scene, background=background).assign_mask("manual", mask)
    inputs = data.inversion_inputs()

    assert set(inputs) == {
        "spectra_targets",
        "spectra_backgrounds",
        "obs_solar_angles",
        "valid_mask",
    }
    assert inputs["spectra_targets"].dtype == np.dtype("float64")
    assert inputs["spectra_backgrounds"].dtype == np.dtype("float64")
    assert inputs["obs_solar_angles"].dtype == np.dtype("float64")

    expected_valid = scene["valid_inversion_mask"] & (~mask)
    expected_valid.name = "valid_inversion_mask"
    xr.testing.assert_identical(inputs["valid_mask"], expected_valid)


def test_cluster_is_reserved():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())

    with pytest.raises(NotImplementedError):
        SpiresData.from_scene(scene).cluster()
