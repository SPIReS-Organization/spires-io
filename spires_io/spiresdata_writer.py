"""Write complete :class:`SpiresData` objects to durable NetCDF products."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import xarray as xr
from spires_contract import SpiresData

from spires_io._spiresdata_netcdf import (
    ANCILLARY_GROUP,
    BACKGROUND_GROUP,
    BACKGROUND_VARIABLE_ATTR,
    NETCDF_ENGINE,
    PRESENT_GROUPS_ATTR,
    PRODUCT_TYPE,
    PRODUCT_TYPE_ATTR,
    RESULTS_GROUP,
    SCENE_GROUP,
    STORAGE_SCHEMA_ATTR,
    STORAGE_SCHEMA_VERSION,
    background_to_dataset,
    prepare_dataset_for_netcdf,
    present_groups,
    validate_product_data,
    validate_product_path,
)

__all__ = ["SpiresDataWriter", "write_spires_data"]


def write_spires_data(
    data: SpiresData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated ``SpiresData`` product.

    The destination is created only after every NetCDF group has been written
    and the temporary file has been flushed. Existing products are preserved
    unless ``overwrite=True`` is explicit.
    """
    output_path = validate_product_path(path)
    validate_product_data(data)

    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"output directory does not exist: {output_path.parent}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    groups = present_groups(data)
    background_dataset = None
    background_variable = None
    if data.background is not None:
        background_dataset, background_variable = background_to_dataset(data.background)

    root_attrs = {
        PRODUCT_TYPE_ATTR: PRODUCT_TYPE,
        STORAGE_SCHEMA_ATTR: STORAGE_SCHEMA_VERSION,
        PRESENT_GROUPS_ATTR: " ".join(groups),
    }
    if background_variable is not None:
        root_attrs[BACKGROUND_VARIABLE_ATTR] = background_variable

    temporary_path = _temporary_output_path(output_path)
    try:
        xr.Dataset(attrs=root_attrs).to_netcdf(
            temporary_path,
            mode="w",
            engine=NETCDF_ENGINE,
            format="NETCDF4",
        )
        _write_group(data.scene, temporary_path, SCENE_GROUP)
        if background_dataset is not None:
            _write_group(background_dataset, temporary_path, BACKGROUND_GROUP)
        if data.ancillary is not None:
            _write_group(data.ancillary, temporary_path, ANCILLARY_GROUP)
        if data.results is not None:
            _write_group(data.results, temporary_path, RESULTS_GROUP)

        _flush_file(temporary_path)
        _promote(temporary_path, output_path, overwrite=overwrite)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path


@dataclass(frozen=True)
class SpiresDataWriter:
    """A writer bound to one ``SpiresData`` object and destination path."""

    data: SpiresData
    output_path: Path
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_path", Path(self.output_path))

    @classmethod
    def from_data(
        cls,
        data: SpiresData,
        *,
        output_path: str | Path,
        overwrite: bool = False,
    ) -> "SpiresDataWriter":
        """Create a writer bound to validated data at write time."""
        return cls(data=data, output_path=Path(output_path), overwrite=overwrite)

    def write(self) -> Path:
        """Write the bound product and return its final path."""
        return write_spires_data(
            self.data,
            self.output_path,
            overwrite=self.overwrite,
        )


def _write_group(dataset: xr.Dataset, path: Path, group: str) -> None:
    prepare_dataset_for_netcdf(dataset).to_netcdf(
        path,
        mode="a",
        group=group,
        engine=NETCDF_ENGINE,
        format="NETCDF4",
    )


def _temporary_output_path(output_path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _flush_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _promote(temporary_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary_path, output_path)
        return

    # A hard link creates the destination atomically without overwriting a file
    # that another process may have created after the initial existence check.
    os.link(temporary_path, output_path)
    temporary_path.unlink()
