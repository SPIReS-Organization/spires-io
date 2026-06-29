"""Path helpers for reusable VIIRS ancillary datasets."""

from pathlib import Path


DEFAULT_STATIC_PRODUCTS = (
    "water_mask",
    "canopy_fraction",
    "dem",
    "glacier_ice_fraction",
    "landcover",
)

DEFAULT_GLOBAL_CATEGORIES = (
    "landcover",
    "canopy",
    "water",
    "dem",
    "glacier_ice",
)


def viirs_sensor_root(base_root: str | Path) -> Path:
    """Return the VIIRS sensor-level data root under a general processing root."""
    return Path(base_root).expanduser().resolve() / "viirs"


def viirs_ancillary_root(base_root: str | Path) -> Path:
    """Return the VIIRS ancillary root under a general processing root."""
    return viirs_sensor_root(base_root) / "ancillary"


def viirs_tile_ancillary_root(base_root: str | Path, tile: str) -> Path:
    """Return the ancillary root for one VIIRS tile."""
    return viirs_ancillary_root(base_root) / "tiles" / tile


def viirs_static_ancillary_path(
    base_root: str | Path,
    tile: str,
    name: str,
    *,
    suffix: str = ".zarr",
) -> Path:
    """Return a static tile-level ancillary path such as canopy or water mask."""
    return viirs_tile_ancillary_root(base_root, tile) / "static" / f"{name}{suffix}"


def viirs_annual_ancillary_path(
    base_root: str | Path,
    tile: str,
    year: int | str,
    name: str,
    *,
    suffix: str = ".zarr",
) -> Path:
    """Return an annual tile-level ancillary path such as R0 or QA summaries."""
    return viirs_tile_ancillary_root(base_root, tile) / "annual" / str(year) / f"{name}{suffix}"


def viirs_ancillary_path(
    base_root: str | Path,
    tile: str,
    name: str,
    *,
    year: int | str | None = None,
    suffix: str = ".zarr",
) -> Path:
    """Return a static or annual VIIRS ancillary path depending on ``year``."""
    if year is None:
        return viirs_static_ancillary_path(base_root, tile, name, suffix=suffix)
    return viirs_annual_ancillary_path(base_root, tile, year, name, suffix=suffix)


def create_viirs_ancillary_layout(
    base_root: str | Path,
    *,
    tiles: tuple[str, ...] | list[str] = (),
    years: tuple[int | str, ...] | list[int | str] = (),
    global_categories: tuple[str, ...] | list[str] = DEFAULT_GLOBAL_CATEGORIES,
) -> dict[str, Path]:
    """
    Create the standard VIIRS ancillary directory layout and return key paths.

    ``base_root`` is the general processing root, for example
    ``examples/outputs`` locally or a shared project data root on a cluster.
    """
    ancillary_root = viirs_ancillary_root(base_root)
    paths = {
        "sensor_root": viirs_sensor_root(base_root),
        "ancillary_root": ancillary_root,
        "global_root": ancillary_root / "global",
        "tiles_root": ancillary_root / "tiles",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    for category in global_categories:
        (paths["global_root"] / category).mkdir(parents=True, exist_ok=True)

    for tile in tiles:
        tile_root = viirs_tile_ancillary_root(base_root, tile)
        (tile_root / "static").mkdir(parents=True, exist_ok=True)
        (tile_root / "annual").mkdir(parents=True, exist_ok=True)
        (tile_root / "logs").mkdir(parents=True, exist_ok=True)
        for year in years:
            (tile_root / "annual" / str(year)).mkdir(parents=True, exist_ok=True)

    return paths
