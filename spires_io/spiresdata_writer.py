"""Write complete :class:`SpiresData` objects to durable NetCDF products."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import xarray as xr
from spires_contract import (
    CONTENT_PROFILE_INVERSION_RAW,
    PersistedProductMetadata,
    PRODUCT_CONTENTS_FULL,
    ProductIdentity,
    SpiresData,
)

from spires_io._spiresdata_netcdf import (
    ANCILLARY_GROUP,
    BACKGROUND_GROUP,
    NETCDF_ENGINE,
    RESULTS_GROUP,
    SCENE_GROUP,
    background_to_dataset,
    build_product_metadata,
    canonical_netcdf_encoding,
    metadata_to_root_attrs,
    prepare_dataset_for_netcdf,
    present_groups,
    product_data_for_contents,
    validate_product_data,
    validate_product_path,
)
from spires_io.persistence_inspection import validate_spires_product

__all__ = ["SpiresDataWriter", "write_spires_data"]


def write_spires_data(
    data: SpiresData,
    path: str | Path,
    *,
    identity: ProductIdentity,
    content_profile: str = CONTENT_PROFILE_INVERSION_RAW,
    product_contents: str = PRODUCT_CONTENTS_FULL,
    completed_operations: tuple[str, ...] = (),
    provenance: Mapping[str, Any] | None = None,
    package_versions: Mapping[str, str] | None = None,
    validation: str = "sample",
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated ``SpiresData`` product.

    The destination is created only after every NetCDF group has been written
    and the temporary file has been flushed. Existing products are preserved
    unless ``overwrite=True`` is explicit.
    """
    output_path = validate_product_path(path)
    stored_data = product_data_for_contents(data, product_contents)
    metadata = build_product_metadata(
        stored_data,
        identity=identity,
        content_profile=content_profile,
        product_contents=product_contents,
        completed_operations=completed_operations,
        provenance=provenance,
        package_versions=package_versions,
    )
    return _write_spires_product(
        stored_data,
        output_path,
        metadata,
        validation=validation,
        overwrite=overwrite,
    )


def _write_spires_product(
    data: SpiresData,
    output_path: Path,
    metadata: PersistedProductMetadata,
    *,
    validation: str,
    overwrite: bool,
    encoding_overrides: Mapping[
        str,
        Mapping[str, Mapping[str, Any]],
    ]
    | None = None,
    expected_existing_state: tuple[int, int, int, int] | None = None,
) -> Path:
    """Write a product with already-constructed metadata.

    This internal entry point is shared with the atomic profile-transition
    implementation so a replacement can preserve the original creation time.
    """
    validate_product_data(data, metadata)

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
        background_dataset, background_variable = background_to_dataset(
            data.background
        )

    root_attrs = metadata_to_root_attrs(
        metadata,
        groups=groups,
        background_variable=background_variable,
    )

    temporary_path = _temporary_output_path(output_path)
    output_mode = _replacement_mode(output_path)
    group_encodings = {} if encoding_overrides is None else encoding_overrides
    try:
        xr.Dataset(attrs=root_attrs).to_netcdf(
            temporary_path,
            mode="w",
            engine=NETCDF_ENGINE,
            format="NETCDF4",
        )
        _write_group(
            data.scene,
            temporary_path,
            SCENE_GROUP,
            encoding_overrides=group_encodings.get(SCENE_GROUP),
        )
        if background_dataset is not None:
            _write_group(
                background_dataset,
                temporary_path,
                BACKGROUND_GROUP,
                encoding_overrides=group_encodings.get(BACKGROUND_GROUP),
            )
        if data.ancillary is not None:
            _write_group(
                data.ancillary,
                temporary_path,
                ANCILLARY_GROUP,
                encoding_overrides=group_encodings.get(ANCILLARY_GROUP),
            )
        if data.results is not None:
            _write_group(
                data.results,
                temporary_path,
                RESULTS_GROUP,
                encoding_overrides=group_encodings.get(RESULTS_GROUP),
            )

        _flush_file(temporary_path)
        validate_spires_product(
            temporary_path,
            expected_identity=metadata.identity,
            expected_profile=metadata.content_profile,
            expected_contents=metadata.product_contents,
            validation=validation,
        )
        temporary_path.chmod(output_mode)
        if expected_existing_state is not None:
            if not output_path.exists():
                raise RuntimeError(
                    "existing product disappeared during atomic update"
                )
            if _file_state(output_path) != expected_existing_state:
                raise RuntimeError(
                    "existing product changed during atomic update"
                )
        _promote(temporary_path, output_path, overwrite=overwrite)
        _flush_directory(output_path.parent)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path


@dataclass(frozen=True)
class SpiresDataWriter:
    """A writer bound to one ``SpiresData`` object and destination path."""

    data: SpiresData
    output_path: Path
    identity: ProductIdentity
    content_profile: str = CONTENT_PROFILE_INVERSION_RAW
    product_contents: str = PRODUCT_CONTENTS_FULL
    completed_operations: tuple[str, ...] = ()
    provenance: Mapping[str, Any] | None = None
    package_versions: Mapping[str, str] | None = None
    validation: str = "sample"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_path", Path(self.output_path))

    @classmethod
    def from_data(
        cls,
        data: SpiresData,
        *,
        output_path: str | Path,
        identity: ProductIdentity,
        content_profile: str = CONTENT_PROFILE_INVERSION_RAW,
        product_contents: str = PRODUCT_CONTENTS_FULL,
        completed_operations: tuple[str, ...] = (),
        provenance: Mapping[str, Any] | None = None,
        package_versions: Mapping[str, str] | None = None,
        validation: str = "sample",
        overwrite: bool = False,
    ) -> "SpiresDataWriter":
        """Create a writer bound to validated data at write time."""
        return cls(
            data=data,
            output_path=Path(output_path),
            identity=identity,
            content_profile=content_profile,
            product_contents=product_contents,
            completed_operations=completed_operations,
            provenance=provenance,
            package_versions=package_versions,
            validation=validation,
            overwrite=overwrite,
        )

    def write(self) -> Path:
        """Write the bound product and return its final path."""
        return write_spires_data(
            self.data,
            self.output_path,
            identity=self.identity,
            content_profile=self.content_profile,
            product_contents=self.product_contents,
            completed_operations=self.completed_operations,
            provenance=self.provenance,
            package_versions=self.package_versions,
            validation=self.validation,
            overwrite=self.overwrite,
        )


def _write_group(
    dataset: xr.Dataset,
    path: Path,
    group: str,
    *,
    encoding_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    prepared = prepare_dataset_for_netcdf(dataset)
    encoding = canonical_netcdf_encoding(prepared)
    if encoding_overrides is not None:
        for name, variable_encoding in encoding_overrides.items():
            if name not in prepared.variables:
                continue
            merged_encoding = encoding.setdefault(name, {})
            merged_encoding.update(variable_encoding)
            if merged_encoding.get("contiguous"):
                for option in (
                    "chunksizes",
                    "zlib",
                    "complevel",
                    "shuffle",
                    "fletcher32",
                ):
                    merged_encoding.pop(option, None)
            elif "chunksizes" in merged_encoding:
                merged_encoding.pop("contiguous", None)
    prepared.to_netcdf(
        path,
        mode="a",
        group=group,
        engine=NETCDF_ENGINE,
        format="NETCDF4",
        encoding=encoding,
    )


def _temporary_output_path(output_path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=f".tmp{output_path.suffix}",
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _flush_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _flush_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replacement_mode(output_path: Path) -> int:
    if output_path.exists():
        return stat.S_IMODE(output_path.stat().st_mode)
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask


def _file_state(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
    )


def _promote(temporary_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary_path, output_path)
        return

    # A hard link creates the destination atomically without overwriting a file
    # that another process may have created after the initial existence check.
    os.link(temporary_path, output_path)
    temporary_path.unlink()
