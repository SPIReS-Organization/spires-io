"""Supported file suffix groups for SPIReS I/O."""

XARRAY_SUFFIXES = frozenset({".nc", ".cdf", ".netcdf"})
RASTER_SUFFIXES = frozenset({".tif", ".tiff"})
HDF_SUFFIXES = frozenset({".hdf", ".h5"})
ZARR_SUFFIXES = frozenset({".zarr"})
MATLAB_SUFFIXES = frozenset({".mat"})

ALL_SUPPORTED_FILE_SUFFIXES = (
    XARRAY_SUFFIXES
    | RASTER_SUFFIXES
    | HDF_SUFFIXES
    | ZARR_SUFFIXES
    | MATLAB_SUFFIXES
)
