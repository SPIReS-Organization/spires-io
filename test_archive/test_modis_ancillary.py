from pathlib import Path

from spires_io.modis.ancillary import (
    create_modis_ancillary_layout,
    modis_annual_ancillary_path,
    modis_ancillary_path,
    modis_ancillary_root,
    modis_sensor_root,
    modis_static_ancillary_path,
    modis_tile_ancillary_root,
)


def test_modis_ancillary_paths_follow_sensor_tile_year_layout(tmp_path):
    base = tmp_path / "project_data"

    assert modis_sensor_root(base) == base.resolve() / "modis"
    assert modis_ancillary_root(base) == base.resolve() / "modis" / "ancillary"
    assert modis_tile_ancillary_root(base, "h08v05") == base.resolve() / "modis" / "ancillary" / "tiles" / "h08v05"
    assert (
        modis_static_ancillary_path(base, "h08v05", "canopy_fraction")
        == base.resolve() / "modis" / "ancillary" / "tiles" / "h08v05" / "static" / "canopy_fraction.zarr"
    )
    assert (
        modis_annual_ancillary_path(base, "h08v05", 2026, "r0_reflectance", suffix=".nc")
        == base.resolve() / "modis" / "ancillary" / "tiles" / "h08v05" / "annual" / "2026" / "r0_reflectance.nc"
    )
    assert modis_ancillary_path(base, "h08v05", "water_mask") == modis_static_ancillary_path(
        base,
        "h08v05",
        "water_mask",
    )
    assert modis_ancillary_path(base, "h08v05", "r0_reflectance", year=2026) == modis_annual_ancillary_path(
        base,
        "h08v05",
        2026,
        "r0_reflectance",
    )


def test_create_modis_ancillary_layout(tmp_path):
    paths = create_modis_ancillary_layout(tmp_path, tiles=["h08v05"], years=[2026])

    expected_dirs = [
        paths["ancillary_root"],
        paths["global_root"] / "canopy",
        paths["global_root"] / "water",
        paths["tiles_root"] / "h08v05" / "static",
        paths["tiles_root"] / "h08v05" / "annual" / "2026",
        paths["tiles_root"] / "h08v05" / "logs",
    ]
    for path in expected_dirs:
        assert Path(path).is_dir()
