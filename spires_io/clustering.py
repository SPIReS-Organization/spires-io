"""Feature-based pixel clustering utilities for SPIReS inputs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


FeatureName = str
RepresentativeMethod = str
Tolerance = Union[float, np.ndarray]

DEFAULT_CLUSTER_FEATURES: tuple[FeatureName, ...] = (
    "reflectance",
    "background",
    "solar_zenith",
)
SUPPORTED_CLUSTER_FEATURES = frozenset(DEFAULT_CLUSTER_FEATURES)


@dataclass(frozen=True)
class ClusteredSpectra:
    """Cluster-local spectra and representative feature values."""

    representative_reflectance: np.ndarray | None
    representative_background: np.ndarray | None
    representative_solar_zenith: np.ndarray | None
    inverse_indices: np.ndarray
    counts: np.ndarray
    valid_flat_indices: np.ndarray
    representative_indices: np.ndarray
    representative_method: RepresentativeMethod
    features: tuple[FeatureName, ...]
    reflectance_tol: np.ndarray | None
    background_tol: np.ndarray | None
    solar_zenith_tol: np.ndarray | None
    original_shape: tuple[int, ...]

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return int(self.counts.size)

    @property
    def n_valid(self) -> int:
        """Number of valid clustered samples."""
        return int(self.valid_flat_indices.size)


def cluster_spectra_rows(
    reflectance: np.ndarray | None = None,
    background: np.ndarray | None = None,
    solar_zenith: np.ndarray | None = None,
    *,
    features: Sequence[FeatureName] | None = None,
    valid_mask: np.ndarray | None = None,
    representative_method: RepresentativeMethod = "cluster_mean",
    tolerance: Tolerance = 0.02,
    reflectance_tol: Tolerance | None = None,
    background_tol: Tolerance | None = None,
    solar_zenith_tol: Tolerance | None = None,
) -> ClusteredSpectra:
    """Cluster rows into approximate unique feature sets."""
    selected_features = _normalize_features(features)
    representative_method = _normalize_representative_method(representative_method)
    arrays = _prepare_row_features(
        selected_features,
        reflectance=reflectance,
        background=background,
        solar_zenith=solar_zenith,
    )
    n_samples = _feature_sample_count(arrays)
    valid = _build_valid_mask(arrays, valid_mask, n_samples)
    valid_flat_indices = np.flatnonzero(valid)
    tolerances = _normalize_feature_tolerances(
        arrays,
        tolerance=tolerance,
        reflectance_tol=reflectance_tol,
        background_tol=background_tol,
        solar_zenith_tol=solar_zenith_tol,
    )

    if valid_flat_indices.size == 0:
        return _empty_clustered_spectra(
            arrays,
            tolerances,
            selected_features,
            representative_method,
            original_shape=_rows_original_shape(arrays, n_samples),
        )

    key_parts = []
    valid_arrays: dict[FeatureName, np.ndarray] = {}
    for feature in selected_features:
        values = arrays[feature][valid]
        valid_arrays[feature] = values
        key_parts.append(_quantize(_as_2d(values), tolerances[feature]))

    key_matrix = np.concatenate(key_parts, axis=1)
    representative_indices, inverse_indices, counts = _row_unique_inverse(key_matrix)
    n_clusters = int(counts.size)
    representatives = _representative_values(
        valid_arrays,
        representative_indices,
        inverse_indices,
        n_clusters,
        counts,
        representative_method,
    )

    return ClusteredSpectra(
        representative_reflectance=representatives.get("reflectance"),
        representative_background=representatives.get("background"),
        representative_solar_zenith=_as_1d_or_none(representatives.get("solar_zenith")),
        inverse_indices=np.ascontiguousarray(inverse_indices),
        counts=np.ascontiguousarray(counts),
        valid_flat_indices=np.ascontiguousarray(valid_flat_indices),
        representative_indices=np.ascontiguousarray(valid_flat_indices[representative_indices]),
        representative_method=representative_method,
        features=selected_features,
        reflectance_tol=tolerances.get("reflectance"),
        background_tol=tolerances.get("background"),
        solar_zenith_tol=tolerances.get("solar_zenith"),
        original_shape=_rows_original_shape(arrays, n_samples),
    )


def cluster_spectra_block(
    reflectance: np.ndarray | None = None,
    background: np.ndarray | None = None,
    solar_zenith: np.ndarray | None = None,
    *,
    features: Sequence[FeatureName] | None = None,
    valid_mask: np.ndarray | None = None,
    representative_method: RepresentativeMethod = "cluster_mean",
    tolerance: Tolerance = 0.02,
    reflectance_tol: Tolerance | None = None,
    background_tol: Tolerance | None = None,
    solar_zenith_tol: Tolerance | None = None,
) -> ClusteredSpectra:
    """Cluster an arbitrary block shaped ``(..., band)`` for spectral features."""
    selected_features = _normalize_features(features)
    block_arrays, sample_shape, n_bands = _prepare_block_features(
        selected_features,
        reflectance=reflectance,
        background=background,
        solar_zenith=solar_zenith,
    )

    flat_reflectance = _flatten_spectral_block(block_arrays.get("reflectance"), n_bands)
    flat_background = _flatten_spectral_block(block_arrays.get("background"), n_bands)
    flat_solar = _flatten_scalar_block(block_arrays.get("solar_zenith"))

    flat_valid_mask = None
    if valid_mask is not None:
        mask = _broadcast_to_shape(valid_mask, sample_shape, "valid_mask", bool)
        flat_valid_mask = np.ascontiguousarray(mask.reshape(-1))

    clustered = cluster_spectra_rows(
        flat_reflectance,
        flat_background,
        flat_solar,
        features=selected_features,
        valid_mask=flat_valid_mask,
        representative_method=representative_method,
        tolerance=tolerance,
        reflectance_tol=reflectance_tol,
        background_tol=background_tol,
        solar_zenith_tol=solar_zenith_tol,
    )

    return ClusteredSpectra(
        representative_reflectance=clustered.representative_reflectance,
        representative_background=clustered.representative_background,
        representative_solar_zenith=clustered.representative_solar_zenith,
        inverse_indices=clustered.inverse_indices,
        counts=clustered.counts,
        valid_flat_indices=clustered.valid_flat_indices,
        representative_indices=clustered.representative_indices,
        representative_method=clustered.representative_method,
        features=clustered.features,
        reflectance_tol=clustered.reflectance_tol,
        background_tol=clustered.background_tol,
        solar_zenith_tol=clustered.solar_zenith_tol,
        original_shape=sample_shape + (n_bands,),
    )


def scatter_cluster_results(
    clustered: ClusteredSpectra,
    cluster_results: np.ndarray,
    *,
    fill_value: float = np.nan,
    n_properties: int | None = None,
) -> np.ndarray:
    """Broadcast cluster-level results back to flattened samples."""
    results = np.asarray(cluster_results, dtype=np.float64)
    if results.ndim == 1:
        results = results[:, None]
    if results.ndim != 2:
        raise ValueError(f"cluster_results must be a 1D or 2D array; got shape {results.shape}")
    if results.shape[0] != clustered.n_clusters:
        raise ValueError(
            "cluster_results first dimension must match the number of clusters; "
            f"got {results.shape[0]} and {clustered.n_clusters}"
        )

    n_samples = int(np.prod(clustered.original_shape[:-1], dtype=np.int64))
    n_properties = results.shape[1] if n_properties is None else int(n_properties)
    if n_properties != results.shape[1]:
        raise ValueError(
            "n_properties must match the second dimension of cluster_results; "
            f"got {n_properties} and {results.shape[1]}"
        )

    full = np.full((n_samples, n_properties), fill_value, dtype=np.float64)
    if clustered.n_valid > 0:
        full[clustered.valid_flat_indices] = results[clustered.inverse_indices]
    return full


def scatter_cluster_results_block(
    clustered: ClusteredSpectra,
    cluster_results: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Broadcast cluster-level results back to the original block shape."""
    flat = scatter_cluster_results(clustered, cluster_results, fill_value=fill_value)
    sample_shape = clustered.original_shape[:-1]
    return flat.reshape(sample_shape + (flat.shape[-1],))


def _normalize_features(features: Sequence[FeatureName] | None) -> tuple[FeatureName, ...]:
    if features is None:
        features = DEFAULT_CLUSTER_FEATURES
    normalized = tuple(str(feature).lower() for feature in features)
    if not normalized:
        raise ValueError("features must include at least one clustering feature")
    unknown = sorted(set(normalized) - SUPPORTED_CLUSTER_FEATURES)
    if unknown:
        raise ValueError(
            "unsupported clustering feature(s): "
            f"{unknown}; supported features are {sorted(SUPPORTED_CLUSTER_FEATURES)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("features must not contain duplicates")
    return normalized


def _normalize_representative_method(method: RepresentativeMethod) -> RepresentativeMethod:
    normalized = method.lower()
    aliases = {"group_mean": "cluster_mean"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"cluster_mean", "first_pixel"}:
        raise ValueError(
            "representative_method must be one of {'cluster_mean', 'first_pixel'}; "
            f"got {method!r}"
        )
    return normalized


def _prepare_row_features(
    features: Sequence[FeatureName],
    *,
    reflectance: np.ndarray | None,
    background: np.ndarray | None,
    solar_zenith: np.ndarray | None,
) -> dict[FeatureName, np.ndarray]:
    arrays: dict[FeatureName, np.ndarray] = {}
    if "reflectance" in features:
        if reflectance is None:
            raise ValueError("reflectance is required when features includes 'reflectance'")
        arrays["reflectance"] = _as_float64_2d(reflectance, "reflectance")
    if "background" in features:
        if background is None:
            raise ValueError("background is required when features includes 'background'")
        arrays["background"] = _as_float64_2d(background, "background")
    if "solar_zenith" in features:
        if solar_zenith is None:
            raise ValueError("solar_zenith is required when features includes 'solar_zenith'")
        arrays["solar_zenith"] = _as_float64_1d(solar_zenith, "solar_zenith")[:, None]

    n_samples = _feature_sample_count(arrays)
    for feature, values in arrays.items():
        if values.shape[0] != n_samples:
            raise ValueError(
                f"{feature} must have the same number of samples as other features; "
                f"got {values.shape[0]} and {n_samples}"
            )
    return arrays


def _prepare_block_features(
    features: Sequence[FeatureName],
    *,
    reflectance: np.ndarray | None,
    background: np.ndarray | None,
    solar_zenith: np.ndarray | None,
) -> tuple[dict[FeatureName, np.ndarray], tuple[int, ...], int]:
    reference = _block_reference_array(features, reflectance, background)
    if reference.ndim < 2:
        raise ValueError(
            "spectral features must have at least one sample dimension plus "
            f"a trailing band dimension; got {reference.shape}"
        )
    sample_shape = reference.shape[:-1]
    n_bands = reference.shape[-1]

    arrays: dict[FeatureName, np.ndarray] = {}
    if "reflectance" in features:
        if reflectance is None:
            raise ValueError("reflectance is required when features includes 'reflectance'")
        arrays["reflectance"] = _broadcast_to_shape(
            reflectance,
            sample_shape + (n_bands,),
            "reflectance",
            np.float64,
        )
    if "background" in features:
        if background is None:
            raise ValueError("background is required when features includes 'background'")
        arrays["background"] = _broadcast_to_shape(
            background,
            sample_shape + (n_bands,),
            "background",
            np.float64,
        )
    if "solar_zenith" in features:
        if solar_zenith is None:
            raise ValueError("solar_zenith is required when features includes 'solar_zenith'")
        arrays["solar_zenith"] = _broadcast_to_shape(
            solar_zenith,
            sample_shape,
            "solar_zenith",
            np.float64,
        )
    return arrays, sample_shape, n_bands


def _block_reference_array(
    features: Sequence[FeatureName],
    reflectance: np.ndarray | None,
    background: np.ndarray | None,
) -> np.ndarray:
    if "reflectance" in features:
        if reflectance is None:
            raise ValueError("reflectance is required when features includes 'reflectance'")
        return np.asarray(reflectance, dtype=np.float64)
    if "background" in features:
        if background is None:
            raise ValueError("background is required when features includes 'background'")
        return np.asarray(background, dtype=np.float64)
    raise ValueError("block clustering requires 'reflectance' or 'background' to define band shape")


def _feature_sample_count(arrays: dict[FeatureName, np.ndarray]) -> int:
    first = next(iter(arrays.values()))
    return int(first.shape[0])


def _rows_original_shape(arrays: dict[FeatureName, np.ndarray], n_samples: int) -> tuple[int, ...]:
    if "reflectance" in arrays:
        return arrays["reflectance"].shape
    if "background" in arrays:
        return arrays["background"].shape
    return (n_samples, 1)


def _build_valid_mask(
    arrays: dict[FeatureName, np.ndarray],
    valid_mask: np.ndarray | None,
    n_samples: int,
) -> np.ndarray:
    finite_mask = np.ones(n_samples, dtype=bool)
    for values in arrays.values():
        finite_mask &= np.all(np.isfinite(values), axis=1)

    if valid_mask is None:
        return finite_mask

    provided = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if provided.shape[0] != n_samples:
        raise ValueError(
            "valid_mask must have the same number of samples as the clustering features; "
            f"got {provided.shape[0]} and {n_samples}"
        )
    return finite_mask & provided


def _normalize_feature_tolerances(
    arrays: dict[FeatureName, np.ndarray],
    *,
    tolerance: Tolerance,
    reflectance_tol: Tolerance | None,
    background_tol: Tolerance | None,
    solar_zenith_tol: Tolerance | None,
) -> dict[FeatureName, np.ndarray]:
    tolerances: dict[FeatureName, np.ndarray] = {}
    if "reflectance" in arrays:
        tolerances["reflectance"] = _normalize_tolerance(
            reflectance_tol,
            tolerance,
            arrays["reflectance"].shape[1],
            "reflectance_tol",
        )
    if "background" in arrays:
        tolerances["background"] = _normalize_tolerance(
            background_tol,
            tolerance,
            arrays["background"].shape[1],
            "background_tol",
        )
    if "solar_zenith" in arrays:
        tolerances["solar_zenith"] = _normalize_tolerance(
            solar_zenith_tol,
            _scale_tolerance(tolerance, 100.0),
            1,
            "solar_zenith_tol",
        )
    return tolerances


def _normalize_tolerance(
    value: Tolerance | None,
    fallback: Tolerance,
    size: int,
    name: str,
) -> np.ndarray:
    base = fallback if value is None else value
    arr = np.asarray(base, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full(size, float(arr), dtype=np.float64)
    elif arr.ndim == 1 and arr.size == size:
        out = arr.astype(np.float64, copy=False)
    else:
        raise ValueError(
            f"{name} must be a scalar or a 1D array of length {size}; got shape {arr.shape}"
        )
    if np.any(out <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return np.ascontiguousarray(out)


def _scale_tolerance(value: Tolerance, factor: float) -> Tolerance:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr) * factor
    return arr * factor


def _representative_values(
    valid_arrays: dict[FeatureName, np.ndarray],
    representative_indices: np.ndarray,
    inverse_indices: np.ndarray,
    n_clusters: int,
    counts: np.ndarray,
    representative_method: RepresentativeMethod,
) -> dict[FeatureName, np.ndarray]:
    representatives = {}
    for feature, values in valid_arrays.items():
        if representative_method == "first_pixel":
            representative = values[representative_indices]
        else:
            representative = _cluster_means(values, inverse_indices, n_clusters, counts)
        representatives[feature] = np.ascontiguousarray(representative)
    return representatives


def _cluster_means(
    values: np.ndarray,
    inverse_indices: np.ndarray,
    n_clusters: int,
    counts: np.ndarray,
) -> np.ndarray:
    means = np.zeros((n_clusters, values.shape[1]), dtype=np.float64)
    np.add.at(means, inverse_indices, values)
    means /= counts[:, None]
    return means


def _empty_clustered_spectra(
    arrays: dict[FeatureName, np.ndarray],
    tolerances: dict[FeatureName, np.ndarray],
    features: tuple[FeatureName, ...],
    representative_method: RepresentativeMethod,
    *,
    original_shape: tuple[int, ...],
) -> ClusteredSpectra:
    return ClusteredSpectra(
        representative_reflectance=_empty_representative(arrays.get("reflectance")),
        representative_background=_empty_representative(arrays.get("background")),
        representative_solar_zenith=_as_1d_or_none(
            _empty_representative(arrays.get("solar_zenith"))
        ),
        inverse_indices=np.empty((0,), dtype=np.int64),
        counts=np.empty((0,), dtype=np.int64),
        valid_flat_indices=np.empty((0,), dtype=np.int64),
        representative_indices=np.empty((0,), dtype=np.int64),
        representative_method=representative_method,
        features=features,
        reflectance_tol=tolerances.get("reflectance"),
        background_tol=tolerances.get("background"),
        solar_zenith_tol=tolerances.get("solar_zenith"),
        original_shape=original_shape,
    )


def _empty_representative(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.empty((0, values.shape[1]), dtype=np.float64)


def _as_float64_2d(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array with shape (n_samples, n_bands); "
            f"got {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _as_float64_1d(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array with shape (n_samples,); got {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values[:, None]
    return values


def _as_1d_or_none(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(values.reshape(-1))


def _quantize(values: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    return np.rint(values / tolerance).astype(np.int64, copy=False)


def _row_unique_inverse(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if keys.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    packed = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    ).reshape(-1)
    _, representative_indices, inverse_indices, counts = np.unique(
        packed,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    return representative_indices, inverse_indices, counts


def _broadcast_to_shape(array: np.ndarray, shape: tuple[int, ...], name: str, dtype) -> np.ndarray:
    arr = np.asarray(array, dtype=dtype)
    if arr.shape == shape:
        return arr
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} must have shape {shape} or be broadcastable to it; got {arr.shape}"
        ) from exc


def _flatten_spectral_block(values: np.ndarray | None, n_bands: int) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(values.reshape(-1, n_bands))


def _flatten_scalar_block(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.ascontiguousarray(values.reshape(-1))
