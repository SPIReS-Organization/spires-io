"""Read serialized SPIReS products back into :class:`SpiresData`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import xarray as xr
from spires_contract import ProductIdentity, SpiresData

from spires_io._spiresdata_netcdf import (
    ANCILLARY_GROUP,
    BACKGROUND_GROUP,
    BACKGROUND_VARIABLE_ATTR,
    NETCDF_ENGINE,
    PRESENT_GROUPS_ATTR,
    RESULTS_GROUP,
    SCENE_GROUP,
    metadata_from_root_attrs,
    parse_present_groups,
    restore_dataset_attrs,
    validate_product_data,
    validate_product_path,
)

__all__ = ["SpiresDataReader", "read_spires_data"]


def read_spires_data(
    path: str | Path,
    *,
    expected_identity: ProductIdentity | None = None,
    expected_profile: str | None = None,
    expected_contents: str | None = None,
) -> SpiresData:
    """Load and validate one serialized ``SpiresData`` NetCDF product."""
    product_path = validate_product_path(path)
    root_attrs = _read_root_attrs(product_path)
    metadata = metadata_from_root_attrs(root_attrs)
    _validate_expected_metadata(
        metadata,
        expected_identity=expected_identity,
        expected_profile=expected_profile,
        expected_contents=expected_contents,
    )
    groups = parse_present_groups(root_attrs[PRESENT_GROUPS_ATTR])

    scene = _read_group(product_path, SCENE_GROUP)
    background = None
    if BACKGROUND_GROUP in groups:
        background_dataset = _read_group(product_path, BACKGROUND_GROUP)
        variable_name = root_attrs.get(BACKGROUND_VARIABLE_ATTR)
        if not isinstance(variable_name, str) or variable_name not in background_dataset:
            raise ValueError(
                "serialized SpiresData background group does not contain its "
                "declared background variable"
            )
        background = background_dataset[variable_name]

    ancillary = (
        _read_group(product_path, ANCILLARY_GROUP)
        if ANCILLARY_GROUP in groups
        else None
    )
    results = (
        _read_group(product_path, RESULTS_GROUP)
        if RESULTS_GROUP in groups
        else None
    )

    data = SpiresData(
        scene=scene,
        background=background,
        ancillary=ancillary,
        results=results,
    )
    validate_product_data(data, metadata)
    return data


@dataclass(frozen=True)
class SpiresDataReader:
    """A reader bound to one serialized SPIReS product."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def read(self) -> SpiresData:
        """Read the bound product."""
        return read_spires_data(self.path)


def _read_root_attrs(path: Path) -> dict[str, object]:
    with xr.open_dataset(path, engine=NETCDF_ENGINE) as root:
        return dict(root.attrs)


def _read_group(path: Path, group: str) -> xr.Dataset:
    try:
        with xr.open_dataset(path, group=group, engine=NETCDF_ENGINE) as dataset:
            loaded = dataset.load()
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"serialized SpiresData product is missing or cannot read group {group!r}"
        ) from exc
    return restore_dataset_attrs(loaded)


def _validate_expected_metadata(
    metadata,
    *,
    expected_identity,
    expected_profile,
    expected_contents,
) -> None:
    if expected_identity is not None and metadata.identity != expected_identity:
        raise ValueError(
            "persisted product identity does not match expected identity: "
            f"found {metadata.identity!r}, expected {expected_identity!r}"
        )
    if (
        expected_profile is not None
        and metadata.content_profile != expected_profile
    ):
        raise ValueError(
            f"persisted product profile is {metadata.content_profile!r}, "
            f"expected {expected_profile!r}"
        )
    if (
        expected_contents is not None
        and metadata.product_contents != expected_contents
    ):
        raise ValueError(
            f"persisted product contents are {metadata.product_contents!r}, "
            f"expected {expected_contents!r}"
        )
