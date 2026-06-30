import numpy as np

from spires_io.modis import prepare_modis_scene_for_inversion
from test_modis_hdf import build_mock_modis_raw_dataset


def test_prepare_modis_scene_for_inversion_decodes_cloud_shadow_and_snow_masks():
    raw = build_mock_modis_raw_dataset()
    raw["state_1km"] = raw["state_1km"] + np.uint16(1) + np.uint16(4) + np.uint16(4096) + np.uint16(8192)

    ds = prepare_modis_scene_for_inversion(raw)

    assert ds["mask_cloud"].dtype == bool
    assert ds["mask_cloud_shadow"].dtype == bool
    assert ds["mask_snow"].dtype == bool
    assert bool(ds["mask_cloud"].all())
    assert bool(ds["mask_cloud_shadow"].all())
    assert bool(ds["mask_snow"].all())
    assert bool((~ds["valid_inversion_mask"]).all())
    assert bool((~ds["valid_r0_mask"]).all())
