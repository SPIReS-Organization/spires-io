import numpy as np
import xarray as xr

from spires_io.viirs import prepare_viirs_scene_for_inversion
from test_viirs_hdf import build_mock_viirs_raw_dataset


def test_prepare_viirs_scene_for_inversion_decodes_qa_cloud_shadow_and_snow_masks():
    raw = build_mock_viirs_raw_dataset()
    raw["qa_qf1"] = raw["qa_qf1"] + np.uint16(8)  # bits 2-3 = 10 -> probably cloudy
    raw["qa_qf2"] = raw["qa_qf2"] + np.uint16(8) + np.uint16(32)  # shadow + snow/ice
    raw["qa_qf7"] = raw["qa_qf7"] + np.uint16(1)  # snow present

    ds = prepare_viirs_scene_for_inversion(raw)

    assert ds["mask_cloud"].dtype == bool
    assert ds["mask_cloud_shadow"].dtype == bool
    assert ds["mask_snow"].dtype == bool
    assert bool(ds["mask_cloud"].all())
    assert bool(ds["mask_cloud_shadow"].all())
    assert bool(ds["mask_snow"].all())
    assert bool((~ds["valid_inversion_mask"]).all())
    assert bool((~ds["valid_r0_mask"]).all())


def test_prepare_viirs_scene_for_inversion_can_override_cloud_masks():
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

    assert bool(ds["mask_cloud_qa"].all())
    assert bool(ds["mask_cloud_shadow_qa"].all())
    assert "mask_cloud_external" in ds
    assert "mask_cloud_shadow_external" in ds
    assert not bool(ds["mask_cloud"].any())
    assert not bool(ds["mask_cloud_shadow"].any())
    assert bool(ds["valid_inversion_mask"].all())
