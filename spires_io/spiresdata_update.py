"""Atomic profile transitions for persisted SPIReS products."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from netCDF4 import Dataset as NetCDFDataset
import xarray as xr
from spires_contract import (
    CONTENT_PROFILE_INVERSION_RAW,
    CONTENT_PROFILE_POSTPROCESSED_RAW,
    ContractError,
    ProductIdentity,
    SpiresData,
)
from spires_contract import conventions as contract

from spires_io._spiresdata_netcdf import (
    PERSISTED_GROUPS,
    build_product_metadata,
)
from spires_io.persistence_inspection import validate_spires_product
from spires_io.spiresdata_reader import read_spires_data
from spires_io.spiresdata_writer import _file_state, _write_spires_product

__all__ = ["update_spires_data_atomically"]


def update_spires_data_atomically(
    path: str | Path,
    results: xr.Dataset,
    *,
    completed_operations: Sequence[str],
    provenance: Mapping[str, Any] | None = None,
    package_versions: Mapping[str, str] | None = None,
    expected_identity: ProductIdentity | None = None,
    expected_contents: str | None = None,
    validation: str = "sample",
) -> Path:
    """Merge postprocessing results and atomically replace one raw product.

    The existing product supplies the immutable scene, optional inputs, base
    inversion results, identity, and storage policy. The replacement becomes a
    ``postprocessed_raw`` product only after it passes the requested validation.
    A concurrent modification detected before promotion aborts the replacement.
    """
    product_path = Path(path)
    if not isinstance(results, xr.Dataset):
        raise TypeError(
            f"results must be an xarray.Dataset, got {type(results).__name__}"
        )

    initial_state = _file_state(product_path)
    inspection = validate_spires_product(
        product_path,
        expected_identity=expected_identity,
        expected_contents=expected_contents,
        validation=validation,
    )
    metadata = inspection.metadata
    if metadata is None:
        raise ContractError("persisted product has no validated metadata")
    if metadata.content_profile not in {
        CONTENT_PROFILE_INVERSION_RAW,
        CONTENT_PROFILE_POSTPROCESSED_RAW,
    }:
        raise ContractError(
            "atomic result updates require an inversion_raw or "
            f"postprocessed_raw product, got {metadata.content_profile!r}"
        )

    preserved_encodings = _read_group_encodings(product_path)
    existing = read_spires_data(
        product_path,
        expected_identity=metadata.identity,
        expected_profile=metadata.content_profile,
        expected_contents=metadata.product_contents,
    )
    merged_results = _merge_results(existing.results, results)
    updated_data = SpiresData(
        scene=existing.scene,
        background=existing.background,
        ancillary=existing.ancillary,
        results=merged_results,
    )

    operations = tuple(
        dict.fromkeys(
            (*metadata.completed_operations, *tuple(completed_operations))
        )
    )
    merged_provenance = dict(metadata.provenance)
    if provenance is not None:
        merged_provenance.update(provenance)
    merged_versions = dict(metadata.package_versions)
    if package_versions is not None:
        merged_versions.update(package_versions)

    updated_metadata = build_product_metadata(
        updated_data,
        identity=metadata.identity,
        content_profile=CONTENT_PROFILE_POSTPROCESSED_RAW,
        product_contents=metadata.product_contents,
        completed_operations=operations,
        provenance=merged_provenance,
        package_versions=merged_versions,
        created_at=metadata.created_at,
    )
    return _write_spires_product(
        updated_data,
        product_path,
        updated_metadata,
        validation=validation,
        overwrite=True,
        encoding_overrides=preserved_encodings,
        expected_existing_state=initial_state,
    )


def _merge_results(
    existing: xr.Dataset | None,
    additions: xr.Dataset,
) -> xr.Dataset:
    if existing is None:
        raise ContractError("atomic update requires existing inversion results")

    violations = []
    for name in contract.SPATIAL_DIMS:
        if name not in additions.coords:
            violations.append(f"updated results are missing coordinate {name!r}")
            continue
        if not additions.coords[name].identical(existing.coords[name]):
            violations.append(
                f"updated results coordinate {name!r} does not exactly match "
                "the persisted product"
            )

    for name in contract.RESULT_VARIABLES:
        if name in additions and not additions[name].identical(existing[name]):
            violations.append(
                f"atomic postprocessing must not change base inversion result {name!r}"
            )

    if violations:
        bullets = "\n".join(f"  - {violation}" for violation in violations)
        raise ContractError(f"atomic_update contract violated:\n{bullets}")

    merged = existing.copy(deep=False)
    for name, array in additions.data_vars.items():
        if name not in contract.RESULT_VARIABLES:
            merged[name] = array
    merged.attrs = {**existing.attrs, **additions.attrs}
    return merged


def _read_group_encodings(
    path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    encodings: dict[str, dict[str, dict[str, Any]]] = {}
    with NetCDFDataset(path, mode="r") as root:
        for group_name in PERSISTED_GROUPS:
            if group_name not in root.groups:
                continue
            group_encodings = {}
            for name, variable in root.groups[group_name].variables.items():
                encoding = _variable_encoding(variable)
                if encoding:
                    group_encodings[name] = encoding
            if group_encodings:
                encodings[group_name] = group_encodings
    return encodings


def _variable_encoding(variable) -> dict[str, Any]:
    encoding: dict[str, Any] = {}
    filters = variable.filters()
    if filters is not None:
        for name in ("zlib", "complevel", "shuffle", "fletcher32"):
            value = filters.get(name)
            if value is not None:
                encoding[name] = value

    chunking = variable.chunking()
    if chunking == "contiguous":
        encoding["contiguous"] = True
        for name in ("zlib", "complevel", "shuffle", "fletcher32"):
            encoding.pop(name, None)
    elif isinstance(chunking, (list, tuple)):
        encoding["chunksizes"] = tuple(int(size) for size in chunking)
        encoding.pop("contiguous", None)
    return encoding
