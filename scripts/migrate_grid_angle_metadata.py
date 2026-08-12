#!/usr/bin/env python3
"""Atomically repair SPIReS grid coordinates and decoded angle metadata."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from typing import Any

from netCDF4 import Dataset as NetCDFDataset
import numpy as np
import xarray as xr

from spires_io._spiresdata_netcdf import NETCDF_ENGINE, spatial_grid_digest
from spires_io.persistence_inspection import validate_spires_product


MIGRATION_NAME = "correct_grid_and_angle_metadata"
MIGRATION_VERSION = 1
MIGRATION_KEY = "archive_migrations"
ANGLE_RANGES = {
    "solar_zenith": (0.0, 180.0),
    "solar_azimuth": (-180.0, 180.0),
    "sensor_zenith": (0.0, 180.0),
    "sensor_azimuth": (-180.0, 180.0),
}
ACTIVE_PACKING_ATTRS = ("scale_factor", "add_offset")
SOURCE_PACKING_ATTRS = (
    "source_scale_factor",
    "source_add_offset",
    "source_valid_range",
    "source_fill_value",
    "source_missing_value",
)
ROOT_GRID_DIGEST_ATTR = "spires_grid_digest"
ROOT_PROVENANCE_ATTR = "spires_provenance"
ROOT_UPDATED_ATTR = "spires_updated_at"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_state(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _variable_signature(root: NetCDFDataset) -> dict[str, dict[str, tuple[Any, ...]]]:
    return {
        group_name: {
            name: (str(variable.dtype), tuple(variable.dimensions), tuple(variable.shape))
            for name, variable in group.variables.items()
        }
        for group_name, group in root.groups.items()
    }


def _parse_geotransform(group) -> tuple[float, float, float, float, float, float]:
    if "spatial_ref" not in group.variables:
        raise ValueError(f"{group.path} is missing spatial_ref")
    spatial_ref = group.variables["spatial_ref"]
    if "GeoTransform" not in spatial_ref.ncattrs():
        raise ValueError(f"{group.path}.spatial_ref is missing GeoTransform")
    value = spatial_ref.getncattr("GeoTransform")
    parts = value.split() if isinstance(value, str) else list(value)
    if len(parts) != 6:
        raise ValueError(f"{group.path}.spatial_ref GeoTransform is not length six")
    transform = tuple(float(item) for item in parts)
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{group.path}.spatial_ref GeoTransform is not finite")
    if transform[1] == 0.0 or transform[5] == 0.0:
        raise ValueError(f"{group.path}.spatial_ref has a zero pixel size")
    tolerance = max(1e-6, max(abs(transform[1]), abs(transform[5])) * 1e-8)
    if abs(transform[2]) > tolerance or abs(transform[4]) > tolerance:
        raise ValueError(f"{group.path} is not a north-up rectilinear grid")
    return transform


def _expected_coordinates(
    transform: tuple[float, float, float, float, float, float],
    *,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    origin_x, pixel_width, _, origin_y, _, pixel_height = transform
    indices_x = np.arange(nx, dtype=np.float64)
    indices_y = np.arange(ny, dtype=np.float64)
    boundary_x = origin_x + indices_x * pixel_width
    boundary_y = origin_y + indices_y * pixel_height
    center_x = origin_x + (indices_x + 0.5) * pixel_width
    center_y = origin_y + (indices_y + 0.5) * pixel_height
    tolerance = max(1e-5, max(abs(pixel_width), abs(pixel_height)) * 1e-7)
    return boundary_x, boundary_y, center_x, center_y, tolerance


def _raw_values(variable) -> np.ndarray:
    variable.set_auto_maskandscale(False)
    return np.asarray(variable[:])


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a JSON string")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must encode a JSON object")
    return parsed


def _has_migration(provenance: dict[str, Any]) -> bool:
    records = provenance.get(MIGRATION_KEY, [])
    if not isinstance(records, list):
        raise ValueError(f"provenance {MIGRATION_KEY!r} must be a list")
    return any(
        isinstance(record, dict)
        and record.get("name") == MIGRATION_NAME
        and record.get("version") == MIGRATION_VERSION
        for record in records
    )


def _inspect_source(path: Path) -> dict[str, Any]:
    with NetCDFDataset(path, mode="r") as root:
        if "scene" not in root.groups:
            raise ValueError("product is missing scene group")
        if ROOT_GRID_DIGEST_ATTR not in root.ncattrs():
            raise ValueError(f"product is missing {ROOT_GRID_DIGEST_ATTR}")
        if ROOT_PROVENANCE_ATTR not in root.ncattrs():
            raise ValueError(f"product is missing {ROOT_PROVENANCE_ATTR}")
        provenance = _json_object(
            root.getncattr(ROOT_PROVENANCE_ATTR),
            label=ROOT_PROVENANCE_ATTR,
        )
        already_migrated = _has_migration(provenance)
        signature = _variable_signature(root)

        scene = root.groups["scene"]
        reference_transform = _parse_geotransform(scene)
        grid_states = set()
        grid_mapping_states = set()
        for group in root.groups.values():
            if "x" not in group.variables and "y" not in group.variables:
                continue
            if "x" not in group.variables or "y" not in group.variables:
                raise ValueError(f"{group.path} has an incomplete x/y coordinate pair")
            transform = _parse_geotransform(group)
            coordinate_derived_transform = (
                reference_transform[0] - 0.5 * reference_transform[1],
                reference_transform[1],
                reference_transform[2],
                reference_transform[3] - 0.5 * reference_transform[5],
                reference_transform[4],
                reference_transform[5],
            )
            transform_tolerance = max(
                1e-5,
                max(abs(reference_transform[1]), abs(reference_transform[5]))
                * 1e-7,
            )
            if np.allclose(
                transform,
                reference_transform,
                rtol=0.0,
                atol=transform_tolerance,
            ):
                grid_mapping_states.add("canonical")
            elif np.allclose(
                transform,
                coordinate_derived_transform,
                rtol=0.0,
                atol=transform_tolerance,
            ):
                grid_mapping_states.add("boundary_labels_interpreted_as_centers")
            else:
                raise ValueError(
                    f"{group.path} GeoTransform matches neither the canonical "
                    "scene grid nor the known boundary-label defect"
                )
            x = _raw_values(group.variables["x"])
            y = _raw_values(group.variables["y"])
            boundary_x, boundary_y, center_x, center_y, tolerance = (
                _expected_coordinates(reference_transform, nx=x.size, ny=y.size)
            )
            if np.allclose(x, boundary_x, rtol=0.0, atol=tolerance) and np.allclose(
                y, boundary_y, rtol=0.0, atol=tolerance
            ):
                grid_states.add("boundary_labels")
            elif np.allclose(x, center_x, rtol=0.0, atol=tolerance) and np.allclose(
                y, center_y, rtol=0.0, atol=tolerance
            ):
                grid_states.add("pixel_centers")
            else:
                raise ValueError(f"{group.path} coordinates match neither bounds nor centers")

        if not grid_states:
            raise ValueError("product contains no spatial coordinate groups")
        if len(grid_states) != 1:
            raise ValueError(f"product groups have mixed grid states: {sorted(grid_states)}")

        angle_digests = {}
        active_angle_states = set()
        has_source_metadata = False
        for name, expected_range in ANGLE_RANGES.items():
            if name not in scene.variables:
                raise ValueError(f"scene is missing required angle {name!r}")
            variable = scene.variables[name]
            if np.dtype(variable.dtype) != np.dtype(np.float32):
                raise ValueError(f"scene.{name} dtype is {variable.dtype}, expected float32")
            values = _raw_values(variable)
            finite = values[np.isfinite(values)]
            if finite.size and (
                finite.min() < expected_range[0] or finite.max() > expected_range[1]
            ):
                raise ValueError(
                    f"scene.{name} stored values are outside physical degree range "
                    f"{expected_range}: [{finite.min()}, {finite.max()}]"
                )
            angle_digests[name] = _array_digest(values)
            attrs = set(variable.ncattrs())
            active = attrs.intersection(ACTIVE_PACKING_ATTRS)
            if active:
                if active != set(ACTIVE_PACKING_ATTRS):
                    raise ValueError(f"scene.{name} has incomplete active packing metadata")
                scale = float(variable.getncattr("scale_factor"))
                offset = float(variable.getncattr("add_offset"))
                if not np.isclose(scale, 0.01, rtol=0.0, atol=1e-8) or not np.isclose(
                    offset, 0.0, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        f"scene.{name} has unexpected scale/offset {scale}, {offset}"
                    )
                active_angle_states.add("packed_metadata")
            else:
                active_angle_states.add("decoded_metadata")
            has_source_metadata = has_source_metadata or bool(
                attrs.intersection(SOURCE_PACKING_ATTRS)
            )

        if len(active_angle_states) != 1:
            raise ValueError(
                f"angle variables have mixed packing states: {sorted(active_angle_states)}"
            )

        return {
            "already_migrated": already_migrated,
            "grid_state": next(iter(grid_states)),
            "grid_mapping_states": sorted(grid_mapping_states),
            "angle_state": next(iter(active_angle_states)),
            "has_source_metadata": has_source_metadata,
            "angle_digests": angle_digests,
            "old_grid_digest": str(root.getncattr(ROOT_GRID_DIGEST_ATTR)),
            "signature": signature,
        }


def _migrate_coordinates_and_angles(path: Path) -> None:
    with NetCDFDataset(path, mode="r+") as root:
        scene = root.groups["scene"]
        canonical_spatial_ref = scene.variables["spatial_ref"]
        canonical_geotransform = canonical_spatial_ref.getncattr("GeoTransform")
        transform = _parse_geotransform(scene)
        for group in root.groups.values():
            if "x" not in group.variables and "y" not in group.variables:
                continue
            x_variable = group.variables["x"]
            y_variable = group.variables["y"]
            _, _, center_x, center_y, _ = _expected_coordinates(
                transform,
                nx=x_variable.size,
                ny=y_variable.size,
            )
            x_variable.set_auto_maskandscale(False)
            y_variable.set_auto_maskandscale(False)
            x_variable[:] = center_x.astype(x_variable.dtype)
            y_variable[:] = center_y.astype(y_variable.dtype)
            group.variables["spatial_ref"].setncattr(
                "GeoTransform",
                canonical_geotransform,
            )

        for name, expected_range in ANGLE_RANGES.items():
            variable = scene.variables[name]
            attrs = set(variable.ncattrs())
            scale = float(variable.getncattr("scale_factor")) if "scale_factor" in attrs else 1.0
            offset = float(variable.getncattr("add_offset")) if "add_offset" in attrs else 0.0
            if "valid_range" in attrs:
                valid_range = np.asarray(variable.getncattr("valid_range"), dtype=np.float32)
                if "scale_factor" in attrs or "add_offset" in attrs:
                    valid_range = valid_range * np.float32(scale) + np.float32(offset)
                variable.setncattr("valid_range", np.sort(valid_range).astype(np.float32))
            else:
                variable.setncattr("valid_range", np.asarray(expected_range, dtype=np.float32))
            for attr in (*ACTIVE_PACKING_ATTRS, *SOURCE_PACKING_ATTRS):
                if attr in variable.ncattrs():
                    variable.delncattr(attr)
        root.sync()


def _new_grid_digest(path: Path) -> str:
    with xr.open_dataset(path, group="scene", engine=NETCDF_ENGINE) as scene:
        return spatial_grid_digest(scene)


def _record_migration(
    path: Path,
    *,
    old_grid_digest: str,
    new_grid_digest: str,
    angle_digests: dict[str, str],
) -> str:
    timestamp = _utc_now()
    with NetCDFDataset(path, mode="r+") as root:
        provenance = _json_object(
            root.getncattr(ROOT_PROVENANCE_ATTR),
            label=ROOT_PROVENANCE_ATTR,
        )
        records = provenance.setdefault(MIGRATION_KEY, [])
        if not isinstance(records, list):
            raise ValueError(f"provenance {MIGRATION_KEY!r} must be a list")
        records.append(
            {
                "name": MIGRATION_NAME,
                "version": MIGRATION_VERSION,
                "timestamp": timestamp,
                "previous_grid_digest": old_grid_digest,
                "grid_digest": new_grid_digest,
                "angle_value_sha256": dict(sorted(angle_digests.items())),
                "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID")
                or os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            }
        )
        root.setncattr(ROOT_GRID_DIGEST_ATTR, new_grid_digest)
        root.setncattr(ROOT_UPDATED_ATTR, timestamp)
        root.setncattr(
            ROOT_PROVENANCE_ATTR,
            json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        )
        root.sync()
    return timestamp


def _validate_migrated_copy(
    path: Path,
    *,
    original: dict[str, Any],
    new_grid_digest: str,
) -> None:
    validate_spires_product(path, validation="sample")
    checked = _inspect_source(path)
    if not checked["already_migrated"]:
        raise ValueError("migration provenance record is missing")
    if checked["grid_state"] != "pixel_centers":
        raise ValueError("coordinates are not canonical pixel centers")
    if checked["grid_mapping_states"] != ["canonical"]:
        raise ValueError("group GeoTransforms do not all match the canonical grid")
    if checked["angle_state"] != "decoded_metadata":
        raise ValueError("angle variables retain active packing metadata")
    if checked["has_source_metadata"]:
        raise ValueError("angle variables retain source packing metadata")
    if checked["angle_digests"] != original["angle_digests"]:
        raise ValueError("stored angle values changed during migration")
    if checked["signature"] != original["signature"]:
        raise ValueError("NetCDF group or variable structure changed during migration")
    if checked["old_grid_digest"] != new_grid_digest:
        raise ValueError("root spatial-grid digest does not match corrected grid")


def _copy_file_metadata(source: Path, target: Path) -> None:
    source_status = source.stat()
    os.chmod(target, source_status.st_mode & 0o7777)
    target_status = target.stat()
    if (target_status.st_uid, target_status.st_gid) != (
        source_status.st_uid,
        source_status.st_gid,
    ):
        os.chown(target, source_status.st_uid, source_status.st_gid)
    for name in os.listxattr(source):
        os.setxattr(target, name, os.getxattr(source, name))


def _verify_file_metadata(source: Path, target: Path) -> None:
    source_status = source.stat()
    target_status = target.stat()
    expected = (
        source_status.st_uid,
        source_status.st_gid,
        source_status.st_mode & 0o7777,
    )
    actual = (
        target_status.st_uid,
        target_status.st_gid,
        target_status.st_mode & 0o7777,
    )
    if actual != expected:
        raise ValueError(f"temporary file ownership/mode {actual} differs from {expected}")
    source_xattrs = {name: os.getxattr(source, name) for name in os.listxattr(source)}
    target_xattrs = {name: os.getxattr(target, name) for name in os.listxattr(target)}
    if source_xattrs != target_xattrs:
        raise ValueError("temporary file extended attributes differ from source")


def migrate_file(path_value: str) -> dict[str, Any]:
    path = Path(path_value).resolve()
    started = time.monotonic()
    temporary_path = None
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        state = _file_state(path)
        original = _inspect_source(path)
        if original["already_migrated"]:
            return {
                "path": str(path),
                "status": "skipped",
                "reason": "migration already recorded",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        if (
            original["grid_state"] == "pixel_centers"
            and original["angle_state"] == "decoded_metadata"
            and not original["has_source_metadata"]
        ):
            return {
                "path": str(path),
                "status": "skipped",
                "reason": "file is already correct",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        if original["grid_state"] != "boundary_labels":
            raise ValueError(f"unexpected source grid state {original['grid_state']!r}")
        if original["angle_state"] != "packed_metadata":
            raise ValueError(f"unexpected source angle state {original['angle_state']!r}")

        handle = tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.{MIGRATION_NAME}-v{MIGRATION_VERSION}.",
            suffix=".tmp.nc",
            delete=False,
        )
        temporary_path = Path(handle.name)
        handle.close()
        shutil.copy2(path, temporary_path)
        _copy_file_metadata(path, temporary_path)
        _verify_file_metadata(path, temporary_path)

        _migrate_coordinates_and_angles(temporary_path)
        new_grid_digest = _new_grid_digest(temporary_path)
        timestamp = _record_migration(
            temporary_path,
            old_grid_digest=original["old_grid_digest"],
            new_grid_digest=new_grid_digest,
            angle_digests=original["angle_digests"],
        )
        _validate_migrated_copy(
            temporary_path,
            original=original,
            new_grid_digest=new_grid_digest,
        )
        _verify_file_metadata(path, temporary_path)

        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        if not path.exists() or _file_state(path) != state:
            raise RuntimeError("source product changed during migration")
        os.replace(temporary_path, path)
        temporary_path = None
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

        return {
            "path": str(path),
            "status": "migrated",
            "timestamp": timestamp,
            "previous_grid_digest": original["old_grid_digest"],
            "grid_digest": new_grid_digest,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "size_bytes": path.stat().st_size,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_manifest_record(stream, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def migrate_paths(paths: list[Path], *, workers: int, manifest: Path) -> int:
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    counts = {"migrated": 0, "skipped": 0, "failed": 0}
    with manifest.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(migrate_file, str(path)): path for path in paths
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                counts[record["status"]] += 1
                _write_manifest_record(stream, record)
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(paths),
                            "counts": counts,
                            "last": record,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    print(json.dumps({"summary": counts, "manifest": str(manifest)}, sort_keys=True))
    return 1 if counts["failed"] else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    files = subcommands.add_parser("files", help="migrate explicit copied products")
    files.add_argument("paths", nargs="+", type=Path)
    files.add_argument("--workers", type=int, default=1)
    files.add_argument("--manifest", type=Path, required=True)

    tile = subcommands.add_parser("tile", help="migrate all raw products in one tile")
    tile.add_argument("tile_directory", type=Path)
    tile.add_argument("--workers", type=int, required=True)
    tile.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "files":
        paths = [path.resolve() for path in args.paths]
    else:
        tile_directory = args.tile_directory.resolve()
        if not tile_directory.is_dir():
            raise FileNotFoundError(tile_directory)
        paths = sorted(tile_directory.glob("*_raw.nc"))
        if not paths:
            raise ValueError(f"no raw products found in {tile_directory}")
    return migrate_paths(paths, workers=args.workers, manifest=args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
