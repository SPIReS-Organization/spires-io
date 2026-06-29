from pathlib import Path

from spires_io.viirs.ancillary import (
    create_viirs_ancillary_layout,
    viirs_annual_ancillary_path,
    viirs_ancillary_path,
    viirs_ancillary_root,
    viirs_sensor_root,
    viirs_static_ancillary_path,
    viirs_tile_ancillary_root,
)


def test_viirs_ancillary_paths_follow_sensor_tile_year_layout(tmp_path):
    base = tmp_path / "project_data"

    assert viirs_sensor_root(base) == base.resolve() / "viirs"
    assert viirs_ancillary_root(base) == base.resolve() / "viirs" / "ancillary"
    assert viirs_tile_ancillary_root(base, "h08v05") == base.resolve() / "viirs" / "ancillary" / "tiles" / "h08v05"
    assert (
        viirs_static_ancillary_path(base, "h08v05", "canopy_fraction")
        == base.resolve() / "viirs" / "ancillary" / "tiles" / "h08v05" / "static" / "canopy_fraction.zarr"
    )
    assert (
        viirs_annual_ancillary_path(base, "h08v05", 2026, "r0_reflectance", suffix=".nc")
        == base.resolve() / "viirs" / "ancillary" / "tiles" / "h08v05" / "annual" / "2026" / "r0_reflectance.nc"
    )
    assert viirs_ancillary_path(base, "h08v05", "water_mask") == viirs_static_ancillary_path(
        base,
        "h08v05",
        "water_mask",
    )
    assert viirs_ancillary_path(base, "h08v05", "r0_reflectance", year=2026) == viirs_annual_ancillary_path(
        base,
        "h08v05",
        2026,
        "r0_reflectance",
    )


def test_create_viirs_ancillary_layout(tmp_path):
    paths = create_viirs_ancillary_layout(tmp_path, tiles=["h08v05"], years=[2026])

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
