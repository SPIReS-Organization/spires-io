import numpy as np
import xarray as xr

from spires_io.masks import decode_inversion_exclusions
from spires_io.viirs import prepare_viirs_scene_for_inversion
from test_viirs_load_surface_reflectance import build_mock_viirs_raw_dataset


def test_prepare_viirs_scene_for_inversion_applies_only_hard_qa_failures():
    raw = build_mock_viirs_raw_dataset()
    raw["qa_qf2"] = raw["qa_qf2"] + np.uint16(32)  # snow/ice
    raw["qa_qf4"] = raw["qa_qf4"] + np.uint16(16)  # bad AOT quality
    raw["qa_qf5"] = raw["qa_qf5"] + np.uint16(252)  # bad M-band SR quality
    raw["qa_qf6"] = raw["qa_qf6"] + np.uint16(63)  # bad I/M-band SR quality
    raw["qa_qf7"] = raw["qa_qf7"] + np.uint16(1)  # snow present

    ds = prepare_viirs_scene_for_inversion(raw)
    exclusions = decode_inversion_exclusions(ds)

    assert bool(ds["valid_inversion_mask"].all())
    assert not bool(exclusions["mask_poor_surface_reflectance_quality"].any())
    assert not any(name.startswith("qa_") for name in ds.data_vars)
    assert "inversion_exclusion_flags" in ds
    assert "inversion_exclusion_assessed" in ds

    bad_sdr = raw.copy(deep=True)
    bad_sdr["qa_qf3"] = bad_sdr["qa_qf3"] + np.uint16(1)  # bad M1 SDR
    bad_sdr_ds = prepare_viirs_scene_for_inversion(bad_sdr)
    bad_sdr_exclusions = decode_inversion_exclusions(bad_sdr_ds)

    assert not bool(bad_sdr_ds["valid_inversion_mask"].any())
    assert bool(
        bad_sdr_exclusions["mask_poor_surface_reflectance_quality"].all()
    )

    missing_aot = raw.copy(deep=True)
    missing_aot["qa_qf4"] = missing_aot["qa_qf4"] + np.uint16(32)
    missing_aot_ds = prepare_viirs_scene_for_inversion(missing_aot)

    assert not bool(missing_aot_ds["valid_inversion_mask"].any())


def test_prepare_viirs_scene_for_inversion_combines_external_cloud_masks():
    raw = build_mock_viirs_raw_dataset()
    raw["qa_qf1"] = raw["qa_qf1"] + np.uint16(8)  # QA says cloudy
    raw["qa_qf2"] = raw["qa_qf2"] + np.uint16(8)  # QA says shadow

    external_masks = xr.Dataset(
        data_vars={
            "mask_cloud": xr.DataArray(
                np.zeros((2, 2), dtype=bool),
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]},
            ),
            "mask_cloud_shadow": xr.DataArray(
                np.zeros((2, 2), dtype=bool),
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [0, 1]},
            ),
        }
    )

    ds = prepare_viirs_scene_for_inversion(raw, cloud_mask_source=external_masks)
    exclusions = decode_inversion_exclusions(ds)

    assert bool(exclusions["mask_cloud"].all())
    assert bool(exclusions["mask_cloud_shadow"].all())
    assert not bool(ds["valid_inversion_mask"].any())
    assert not any(name.startswith("qa_") for name in ds.data_vars)
