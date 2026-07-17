import numpy as np
import pytest
import xarray as xr

from spires_io import SpiresData
from spires_io.modis import prepare_modis_scene_for_inversion
from spires_io.viirs import prepare_viirs_scene_for_inversion
from test_modis_load_surface_reflectance import build_mock_modis_raw_dataset
from test_viirs_load_surface_reflectance import build_mock_viirs_raw_dataset


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


def _cluster_scene(
    reflectance=None,
    solar_zenith=None,
    valid_mask=None,
):
    if reflectance is None:
        reflectance = np.array(
            [
                [[0.2, 0.3], [0.2, 0.3]],
                [[0.8, 0.9], [0.2, 0.3]],
            ],
            dtype=np.float32,
        )
    if solar_zenith is None:
        solar_zenith = np.full((2, 2), 30.0, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.array([[True, True], [True, False]], dtype=bool)

    return xr.Dataset(
        {
            "reflectance": xr.DataArray(
                reflectance,
                dims=("y", "x", "band"),
                coords={"y": [0, 1], "x": [10, 11], "band": ["a", "b"]},
                name="reflectance",
            ),
            "solar_zenith": xr.DataArray(
                solar_zenith,
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [10, 11]},
                name="solar_zenith",
            ),
            "valid_inversion_mask": xr.DataArray(
                valid_mask,
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [10, 11]},
                name="valid_inversion_mask",
            ),
        }
    )


def _cluster_background(scene, values=None):
    if values is None:
        values = np.ones(scene["reflectance"].shape, dtype=np.float32)
    return xr.DataArray(
        values,
        dims=("y", "x", "band"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x", "band")},
        name="background_reflectance",
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


def test_assign_viewable_canopy_fraction_preserves_raw_canopy_fraction():
    scene = _cluster_scene()
    scene["sensor_zenith"] = xr.DataArray(
        [[0.0, 10.0], [20.0, 30.0]],
        dims=("y", "x"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x")},
        name="sensor_zenith",
    )
    scene["sensor_azimuth"] = xr.DataArray(
        [[90.0, 90.0], [180.0, 180.0]],
        dims=("y", "x"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x")},
        name="sensor_azimuth",
    )
    canopy_fraction = xr.DataArray(
        [[0.25, 0.25], [0.50, 0.50]],
        dims=("y", "x"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x")},
        name="canopy_fraction",
    )
    slope = xr.DataArray(
        [[0.0, 5.0], [10.0, 15.0]],
        dims=("y", "x"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x")},
        name="slope",
    )
    aspect = xr.DataArray(
        [[0.0, 45.0], [90.0, 135.0]],
        dims=("y", "x"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x")},
        name="aspect",
    )
    ancillary = xr.Dataset(
        {
            "canopy_fraction": canopy_fraction,
            "slope": slope,
            "aspect": aspect,
        }
    )

    data = SpiresData.from_scene(scene, ancillary=ancillary)
    updated = data.assign_viewable_canopy_fraction(
        average_vertical_crown_radius=4.644,
        average_horizontal_crown_radius=1.72,
    )

    assert "viewable_canopy_fraction" not in data.ancillary
    xr.testing.assert_identical(updated.ancillary["canopy_fraction"], canopy_fraction)
    assert updated.ancillary["viewable_canopy_fraction"].dims == ("y", "x")

    b_r = 4.644 / 1.72
    theta_v_prime = np.arctan(b_r * np.tan(np.deg2rad(scene["sensor_zenith"])))
    theta_s_prime = np.deg2rad(
        90.0 - np.rad2deg(np.arctan(b_r * np.tan(np.deg2rad(90.0 - slope))))
    )
    phi_v_prime = np.deg2rad(scene["sensor_azimuth"] - aspect)
    exponent = np.cos(theta_s_prime) / (
        np.cos(phi_v_prime) * np.sin(theta_v_prime) * np.sin(theta_s_prime)
        + np.cos(theta_v_prime) * np.cos(theta_s_prime)
    )
    expected = (1.0 - ((1.0 - canopy_fraction) ** exponent)).astype("float32")
    expected.name = "viewable_canopy_fraction"
    xr.testing.assert_allclose(
        updated.ancillary["viewable_canopy_fraction"],
        expected,
        rtol=1e-6,
    )


def test_inversion_inputs_requires_background():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())

    with pytest.raises(ValueError, match="background must be assigned"):
        SpiresData.from_scene(scene).inversion_inputs()


def test_inversion_inputs_returns_float32_arrays_and_updated_valid_mask():
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
    # io->inversion boundary is float32 (spires-contract ACCEPTED_DTYPES).
    assert inputs["spectra_targets"].dtype == np.dtype("float32")
    assert inputs["spectra_backgrounds"].dtype == np.dtype("float32")
    assert inputs["obs_solar_angles"].dtype == np.dtype("float32")

    expected_valid = scene["valid_inversion_mask"] & (~mask)
    expected_valid.name = "valid_inversion_mask"
    xr.testing.assert_identical(inputs["valid_mask"], expected_valid)


def test_default_cluster_requires_background():
    scene = prepare_modis_scene_for_inversion(build_mock_modis_raw_dataset())

    with pytest.raises(ValueError, match="background is required"):
        SpiresData.from_scene(scene).cluster()


def test_cluster_reflectance_only_stores_labels_counts_and_representatives():
    data = SpiresData.from_scene(_cluster_scene())

    clustered = data.cluster(features=("reflectance",), representative_method="first_pixel")

    assert clustered is not data
    assert "cluster_label" not in data.scene
    assert "cluster_label" in clustered.scene
    assert "cluster_count" in clustered.scene
    assert "cluster_representative_reflectance" in clustered.scene
    assert "cluster_representative_background" not in clustered.scene
    assert "cluster_representative_solar_zenith" not in clustered.scene

    labels = clustered.scene["cluster_label"]
    assert labels.sel(y=0, x=10).item() == labels.sel(y=0, x=11).item()
    assert labels.sel(y=1, x=10).item() != labels.sel(y=0, x=10).item()
    assert labels.sel(y=1, x=11).item() == -1
    np.testing.assert_array_equal(clustered.scene["cluster_count"].values, np.array([2, 1]))
    assert clustered.scene["cluster_representative_reflectance"].dims == ("cluster", "band")


def test_cluster_uses_cluster_mean_representatives():
    reflectance = np.array(
        [
            [[0.201, 0.301], [0.209, 0.309]],
            [[0.7, 0.8], [0.9, 1.0]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([[True, True], [False, False]], dtype=bool)
    data = SpiresData.from_scene(
        _cluster_scene(reflectance=reflectance, valid_mask=valid_mask)
    )

    clustered = data.cluster(features=("reflectance",), reflectance_tol=0.05)

    np.testing.assert_allclose(
        clustered.scene["cluster_representative_reflectance"].values,
        np.array([[0.205, 0.305]], dtype=np.float64),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(clustered.scene["cluster_count"].values, np.array([2]))


def test_cluster_uses_configured_defaults_when_args_omitted():
    data = SpiresData.from_scene(
        _cluster_scene(valid_mask=np.ones((2, 2), dtype=bool)),
        cluster_defaults={
            "features": ("reflectance",),
            "label_name": "configured_cluster",
            "representative_method": "first_pixel",
            "reflectance_tol": 0.05,
        },
    )

    clustered = data.cluster()

    assert "configured_cluster" in clustered.scene
    assert clustered.scene["configured_cluster"].attrs["features"] == "reflectance"
    assert (
        clustered.scene["cluster_representative_reflectance"]
        .attrs["representative_method"]
        == "first_pixel"
    )


def test_cluster_default_features_store_background_and_solar_representatives():
    scene = _cluster_scene()
    background = _cluster_background(scene)
    data = SpiresData.from_scene(scene, background=background)

    clustered = data.cluster()

    assert "cluster_representative_reflectance" in clustered.scene
    assert "cluster_representative_background" in clustered.scene
    assert "cluster_representative_solar_zenith" in clustered.scene
    assert clustered.scene["cluster_label"].attrs["features"] == (
        "reflectance,background,solar_zenith"
    )


def test_cluster_background_feature_affects_labels():
    reflectance = np.full((2, 2, 2), 0.2, dtype=np.float32)
    background = np.array(
        [
            [[0.1, 0.1], [0.4, 0.4]],
            [[0.1, 0.1], [0.4, 0.4]],
        ],
        dtype=np.float32,
    )
    scene = _cluster_scene(reflectance=reflectance, valid_mask=np.ones((2, 2), dtype=bool))
    data = SpiresData.from_scene(scene, background=_cluster_background(scene, background))

    reflectance_only = data.cluster(features=("reflectance",))
    with_background = data.cluster(features=("reflectance", "background"))

    assert reflectance_only.scene.sizes["cluster"] == 1
    assert with_background.scene.sizes["cluster"] == 2
    assert (
        with_background.scene["cluster_label"].sel(y=0, x=10).item()
        != with_background.scene["cluster_label"].sel(y=0, x=11).item()
    )


def test_cluster_solar_feature_affects_labels():
    reflectance = np.full((2, 2, 2), 0.2, dtype=np.float32)
    solar = np.array([[30.0, 40.0], [30.0, 40.0]], dtype=np.float32)
    scene = _cluster_scene(
        reflectance=reflectance,
        solar_zenith=solar,
        valid_mask=np.ones((2, 2), dtype=bool),
    )
    data = SpiresData.from_scene(scene)

    reflectance_only = data.cluster(features=("reflectance",))
    with_solar = data.cluster(
        features=("reflectance", "solar_zenith"),
        solar_zenith_tol=1.0,
    )

    assert reflectance_only.scene.sizes["cluster"] == 1
    assert with_solar.scene.sizes["cluster"] == 2
    assert (
        with_solar.scene["cluster_label"].sel(y=0, x=10).item()
        != with_solar.scene["cluster_label"].sel(y=0, x=11).item()
    )


def test_cluster_preserves_background_ancillary_and_original_data():
    scene = _cluster_scene()
    background = _cluster_background(scene)
    ancillary = xr.Dataset(
        {
            "dem": xr.DataArray(
                np.ones((2, 2), dtype=np.float32),
                dims=("y", "x"),
                coords={dim: scene.coords[dim].values for dim in ("y", "x")},
            )
        }
    )
    data = SpiresData.from_scene(scene, background=background, ancillary=ancillary)

    clustered = data.cluster(features=("reflectance",), label_name="my_cluster_label")

    assert "my_cluster_label" not in data.scene
    assert "my_cluster_label" in clustered.scene
    assert clustered.background_spectra.identical(background)
    assert clustered.ancillary.identical(ancillary)
