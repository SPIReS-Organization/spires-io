"""Feature-based pixel clustering utilities for SPIReS inputs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Union

import numpy as np


FeatureName = str
RepresentativeMethod = str
Tolerance = Union[float, np.ndarray]
FeatureKind = Literal["spectral", "scalar"]


@dataclass(frozen=True)
class ClusterFeatureSpec:
    """Shape and default-tolerance policy for one clustering feature."""

    kind: FeatureKind
    default_tolerance: float
    representative_long_name: str | None = None
    units: str | None = None


CLUSTER_FEATURE_SPECS: Mapping[FeatureName, ClusterFeatureSpec] = MappingProxyType(
    {
        "reflectance": ClusterFeatureSpec("spectral", 0.02),
        "background": ClusterFeatureSpec("spectral", 0.02),
        "solar_zenith": ClusterFeatureSpec("scalar", 2.0),
        "cosine_illumination": ClusterFeatureSpec(
            "scalar",
            0.02,
            representative_long_name=(
                "Representative cosine of local solar incidence"
            ),
            units="1",
        ),
    }
)

DEFAULT_CLUSTER_FEATURES: tuple[FeatureName, ...] = (
    "reflectance",
    "background",
    "solar_zenith",
)
SUPPORTED_CLUSTER_FEATURES = frozenset(CLUSTER_FEATURE_SPECS)


@dataclass(frozen=True)
class ClusteredSpectra:
    """Cluster-local spectra and representative feature values."""

    representative_reflectance: np.ndarray | None
    representative_background: np.ndarray | None
    representative_solar_zenith: np.ndarray | None
    representative_cosine_illumination: np.ndarray | None
    inverse_indices: np.ndarray
    counts: np.ndarray
    valid_flat_indices: np.ndarray
    representative_indices: np.ndarray
    representative_method: RepresentativeMethod
    features: tuple[FeatureName, ...]
    reflectance_tol: np.ndarray | None
    background_tol: np.ndarray | None
    solar_zenith_tol: np.ndarray | None
    cosine_illumination_tol: np.ndarray | None
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
    cosine_illumination: np.ndarray | None = None,
    *,
    features: Sequence[FeatureName] | None = None,
    valid_mask: np.ndarray | None = None,
    representative_method: RepresentativeMethod = "cluster_mean",
    reflectance_tol: Tolerance = (
        CLUSTER_FEATURE_SPECS["reflectance"].default_tolerance
    ),
    background_tol: Tolerance = (
        CLUSTER_FEATURE_SPECS["background"].default_tolerance
    ),
    solar_zenith_tol: Tolerance = (
        CLUSTER_FEATURE_SPECS["solar_zenith"].default_tolerance
    ),
    cosine_illumination_tol: Tolerance = CLUSTER_FEATURE_SPECS[
        "cosine_illumination"
    ].default_tolerance,
) -> ClusteredSpectra:
    """Cluster rows into approximate unique feature sets."""
    selected_features = _normalize_features(features)
    representative_method = _normalize_representative_method(representative_method)
    arrays = _prepare_row_features(
        selected_features,
        _feature_values(
            reflectance=reflectance,
            background=background,
            solar_zenith=solar_zenith,
            cosine_illumination=cosine_illumination,
        ),
    )
    n_samples = _feature_sample_count(arrays)
    valid = _build_valid_mask(arrays, valid_mask, n_samples)
    valid_flat_indices = np.flatnonzero(valid)
    tolerances = _normalize_feature_tolerances(
        arrays,
        _feature_tolerances(
            reflectance_tol=reflectance_tol,
            background_tol=background_tol,
            solar_zenith_tol=solar_zenith_tol,
            cosine_illumination_tol=cosine_illumination_tol,
        ),
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
        representative_cosine_illumination=_as_1d_or_none(
            representatives.get("cosine_illumination")
        ),
        inverse_indices=np.ascontiguousarray(inverse_indices),
        counts=np.ascontiguousarray(counts),
        valid_flat_indices=np.ascontiguousarray(valid_flat_indices),
        representative_indices=np.ascontiguousarray(valid_flat_indices[representative_indices]),
        representative_method=representative_method,
        features=selected_features,
        reflectance_tol=tolerances.get("reflectance"),
        background_tol=tolerances.get("background"),
        solar_zenith_tol=tolerances.get("solar_zenith"),
        cosine_illumination_tol=tolerances.get("cosine_illumination"),
        original_shape=_rows_original_shape(arrays, n_samples),
    )


def cluster_spectra_block(
    reflectance: np.ndarray | None = None,
    background: np.ndarray | None = None,
    solar_zenith: np.ndarray | None = None,
    cosine_illumination: np.ndarray | None = None,
    *,
    features: Sequence[FeatureName] | None = None,
    valid_mask: np.ndarray | None = None,
    representative_method: RepresentativeMethod = "cluster_mean",
    reflectance_tol: Tolerance = (
        CLUSTER_FEATURE_SPECS["reflectance"].default_tolerance
    ),
    background_tol: Tolerance = (
        CLUSTER_FEATURE_SPECS["background"].default_tolerance
    ),
    solar_zenith_tol: Tolerance = (
        CLUSTER_FEATURE_SPECS["solar_zenith"].default_tolerance
    ),
    cosine_illumination_tol: Tolerance = CLUSTER_FEATURE_SPECS[
        "cosine_illumination"
    ].default_tolerance,
) -> ClusteredSpectra:
    """Cluster an arbitrary block shaped ``(..., band)`` for spectral features."""
    selected_features = _normalize_features(features)
    block_arrays, sample_shape, n_bands = _prepare_block_features(
        selected_features,
        _feature_values(
            reflectance=reflectance,
            background=background,
            solar_zenith=solar_zenith,
            cosine_illumination=cosine_illumination,
        ),
    )

    flat_arrays = {
        feature: (
            _flatten_spectral_block(values, n_bands)
            if CLUSTER_FEATURE_SPECS[feature].kind == "spectral"
            else _flatten_scalar_block(values)
        )
        for feature, values in block_arrays.items()
    }

    flat_valid_mask = None
    if valid_mask is not None:
        mask = _broadcast_to_shape(valid_mask, sample_shape, "valid_mask", bool)
        flat_valid_mask = np.ascontiguousarray(mask.reshape(-1))

    clustered = cluster_spectra_rows(
        **flat_arrays,
        features=selected_features,
        valid_mask=flat_valid_mask,
        representative_method=representative_method,
        reflectance_tol=reflectance_tol,
        background_tol=background_tol,
        solar_zenith_tol=solar_zenith_tol,
        cosine_illumination_tol=cosine_illumination_tol,
    )

    return replace(clustered, original_shape=sample_shape + (n_bands,))


def scatter_cluster_results(
    clustered: ClusteredSpectra,
    cluster_results: np.ndarray,
    *,
    fill_value: float = np.nan,
    n_properties: int | None = None,
) -> np.ndarray:
    """Broadcast cluster-level results back to flattened samples."""
    results = np.asarray(cluster_results, dtype=np.float32)
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

    full = np.full((n_samples, n_properties), fill_value, dtype=np.float32)
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


def _feature_values(
    *,
    reflectance: np.ndarray | None,
    background: np.ndarray | None,
    solar_zenith: np.ndarray | None,
    cosine_illumination: np.ndarray | None,
) -> dict[FeatureName, np.ndarray | None]:
    return {
        "reflectance": reflectance,
        "background": background,
        "solar_zenith": solar_zenith,
        "cosine_illumination": cosine_illumination,
    }


def _feature_tolerances(
    *,
    reflectance_tol: Tolerance,
    background_tol: Tolerance,
    solar_zenith_tol: Tolerance,
    cosine_illumination_tol: Tolerance,
) -> dict[FeatureName, Tolerance]:
    return {
        "reflectance": reflectance_tol,
        "background": background_tol,
        "solar_zenith": solar_zenith_tol,
        "cosine_illumination": cosine_illumination_tol,
    }


def _prepare_row_features(
    features: Sequence[FeatureName],
    supplied: dict[FeatureName, np.ndarray | None],
) -> dict[FeatureName, np.ndarray]:
    arrays: dict[FeatureName, np.ndarray] = {}
    for feature in features:
        values = supplied[feature]
        if values is None:
            raise ValueError(
                f"{feature} is required when features includes {feature!r}"
            )
        if CLUSTER_FEATURE_SPECS[feature].kind == "spectral":
            arrays[feature] = _as_float32_2d(values, feature)
        else:
            arrays[feature] = _as_float32_1d(values, feature)[:, None]

    n_samples = _feature_sample_count(arrays)
    for feature, values in arrays.items():
        if values.shape[0] not in {1, n_samples}:
            raise ValueError(
                f"{feature} must have {n_samples} samples or one broadcastable "
                f"sample; got {values.shape[0]}"
            )
        if values.shape[0] == 1 and n_samples != 1:
            arrays[feature] = np.broadcast_to(
                values, (n_samples, values.shape[1])
            )
    return arrays


def _prepare_block_features(
    features: Sequence[FeatureName],
    supplied: dict[FeatureName, np.ndarray | None],
) -> tuple[dict[FeatureName, np.ndarray], tuple[int, ...], int]:
    sample_shapes: list[tuple[int, ...]] = []
    spectral_band_counts: list[int] = []
    for feature in features:
        value = supplied[feature]
        if value is None:
            raise ValueError(f"{feature} is required when features includes {feature!r}")
        shape = np.shape(value)
        if CLUSTER_FEATURE_SPECS[feature].kind == "spectral":
            if len(shape) < 2:
                raise ValueError(
                    "spectral features must have at least one sample dimension plus "
                    f"a trailing band dimension; got {shape}"
                )
            sample_shapes.append(shape[:-1])
            spectral_band_counts.append(shape[-1])
        else:
            if not shape:
                raise ValueError(f"{feature} must have at least one sample dimension")
            sample_shapes.append(shape)

    try:
        sample_shape = np.broadcast_shapes(*sample_shapes)
    except ValueError as exc:
        raise ValueError(
            "selected clustering features have incompatible sample shapes: "
            f"{sample_shapes}"
        ) from exc
    if spectral_band_counts and len(set(spectral_band_counts)) != 1:
        raise ValueError(
            "selected spectral clustering features must have the same number "
            f"of bands; got {spectral_band_counts}"
        )
    n_bands = spectral_band_counts[0] if spectral_band_counts else 1

    arrays = {
        feature: _broadcast_to_shape(
            supplied[feature],
            (
                sample_shape + (n_bands,)
                if CLUSTER_FEATURE_SPECS[feature].kind == "spectral"
                else sample_shape
            ),
            feature,
            np.float32,
        )
        for feature in features
    }
    return arrays, sample_shape, n_bands


def _feature_sample_count(arrays: dict[FeatureName, np.ndarray]) -> int:
    return max(int(values.shape[0]) for values in arrays.values())


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
    supplied: dict[FeatureName, Tolerance],
) -> dict[FeatureName, np.ndarray]:
    return {
        feature: _normalize_tolerance(
            supplied[feature],
            values.shape[1],
            f"{feature}_tol",
        )
        for feature, values in arrays.items()
    }


def _normalize_tolerance(
    value: Tolerance,
    size: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        out = np.full(size, float(arr), dtype=np.float32)
    elif arr.ndim == 1 and arr.size == size:
        out = arr.astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"{name} must be a scalar or a 1D array of length {size}; got shape {arr.shape}"
        )
    if np.any(out <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return np.ascontiguousarray(out)


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
        representatives[feature] = np.ascontiguousarray(
            representative, dtype=np.float32
        )
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
        representative_cosine_illumination=_as_1d_or_none(
            _empty_representative(arrays.get("cosine_illumination"))
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
        cosine_illumination_tol=tolerances.get("cosine_illumination"),
        original_shape=original_shape,
    )


def _empty_representative(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.empty((0, values.shape[1]), dtype=np.float32)


def _as_float32_2d(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array with shape (n_samples, n_bands); "
            f"got {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _as_float32_1d(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
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
