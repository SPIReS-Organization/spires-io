# spires-io

I/O for the [SPIReS](https://github.com/SPIReS-Organization) package family:
MODIS / Sentinel-2 / Landsat loaders, reprojection, and coordinate transforms.

Produces target/background reflectance spectra and solar angles that conform to
the I/O→inversion boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract).
The package imports and re-exports the neutral `SpiresData` container from
`spires-contract`; loading, masking, and clustering remain package-level I/O
operations rather than methods on that shared object.

```python
import spires_io

data = spires_io.load("config.json")
data = spires_io.cluster(data, features=("reflectance", "solar_zenith"))
```

## Serialized SpiresData products

Complete inversion and postprocessing outputs can be written atomically to a
grouped NetCDF product and reconstructed for a later scientific stage:

```python
from spires_contract import ProductIdentity
import spires_io

identity = ProductIdentity(
    sensor="viirs",
    platform="noaa20",
    product="vj109ga",
    spatial_id="h09v04",
    acquisition_time="2026-03-14",
)
output_path = spires_io.write_spires_data(
    data,
    "spires_vj109ga_h09v04_20260314_raw.nc",
    identity=identity,
    content_profile="inversion_raw",
    product_contents="full",
)
restored = spires_io.read_spires_data(output_path)
```

The root records schema version 1 identity, completion, grid digest, package
versions, timestamps, and provenance. The `scene`, `background`, `ancillary`,
and `results` fields use matching NetCDF groups when present. New files use
lossless compression and are promoted only after the temporary product has
been reopened and validated. Existing destinations are preserved unless
overwrite is explicit.

MODIS and VIIRS HDF-EOS loaders derive canonical `x` and `y` pixel centers
from the declared grid bounds, dimensions, and resolution. Persistence
validation requires those centers to agree with the `spatial_ref`
`GeoTransform`; dimension-scale values that identify pixel boundaries cannot
be persisted as coordinate centers.

Completion profile and stored payload are separate:

- `inversion_raw` contains completed base inversion output.
- `postprocessed_raw` contains base output plus its declared completed
  operations.
- `full` stores the complete inversion-ready object and results.
- `results_subset` stores only the self-describing spatial grid, complete
  packed QA, and results.

A fused inversion/postprocessing workflow can write directly as
`postprocessed_raw`. A piecewise workflow first writes `inversion_raw`, then
merges derived results into an atomically replaced `postprocessed_raw` file:

```python
spires_io.update_spires_data_atomically(
    output_path,
    postprocessed.results,
    completed_operations=("canopy_correction", "albedo"),
)
```

Atomic updates retain the original identity, creation time, payload choice,
inputs, base inversion results, valid unknown result variables, file
permissions, and existing variable encodings. A changed base result, grid
mismatch, validation failure, or concurrent file replacement aborts the update
without replacing the original.

Use lightweight inspection before loading scientific arrays:

```python
inspection = spires_io.inspect_spires_product(
    output_path,
    expected_identity=identity,
    expected_profile="postprocessed_raw",
)
assert inspection.complete
```

`validate_spires_product()` supports `metadata`, `sample`, and `full`
validation levels; `sample` is the default for writes and atomic updates.

## Inversion-exclusion provenance

Prepared MODIS and VIIRS scenes store inversion exclusions in the canonical
`uint16` pair `inversion_exclusion_flags` and
`inversion_exclusion_assessed`, together with `valid_inversion_mask`. Use
`decode_inversion_exclusions()` to regenerate Boolean reason and assessed-state
masks. `write_detailed_masks` has been removed because the packed pair is the
lossless, canonical representation.

For VIIRS, selected-band bad-SDR flags from QF3-QF4 and missing or invalid
atmospheric-correction inputs from QF4-QF5 contribute to the
`poor_surface_reflectance_quality` exclusion. QF4 AOT quality and the QF5-QF6
overall surface-reflectance-quality bits are not inversion exclusions because
they are systematically set over snow. QA fields are decoded transiently:
raw, stacked, and decoded QA arrays are omitted from the prepared inversion
scene after the SPIRES-owned `inversion_exclusion_flags`,
`inversion_exclusion_assessed`, and `valid_inversion_mask` are constructed.
Those three canonical mask variables remain in the prepared scene.

Cloud-mask provenance is controlled independently from when cloud exclusions
are applied:

```json
{
  "mask": {
    "cloud_mask_source_policy": "external_only",
    "cloud_mask_application_stage": "pre_inversion"
  }
}
```

The source policy accepts `external_only`, `qa_only`, `qa_or_external`, or
`none`. An external-only policy requires `files.cloud_mask` and does not add
native QA cloud or cloud-shadow flags. Omitting the policy restores the
historical selection: QA-only without an external mask and external-only when
one is supplied. The application stage accepts
`pre_inversion` or `post_inversion`; the latter leaves cloud reasons
unassessed in the inversion-exclusion flags so an independently persisted
cloud mask can be applied downstream.

## Master products and band selection

`sensor.selected_bands` is the explicit, ordered down-selection for a sensor
master LUT and other all-band inputs:

```json
{
  "sensor": {
    "name": "viirs",
    "selected_bands": ["I1", "I2", "M4", "M7"]
  }
}
```

The prepared scene preserves this order in its `band` coordinate. Labeled
all-band background products are selected and reordered to match it, and
downstream `spires-inversion` should use that coordinate to select the same
bands from the numerical reflectance LUT. `spires-io` does not load LUT
values. While legacy MATLAB LUTs remain supported, omitting
`sensor.selected_bands` still allows their metadata or filename to provide a
transitional band selection; explicit configuration always takes precedence.

## Inversion runtime configuration

The `inversion` section configures the high-level
`spires_inversion.invert()` object API:

```json
{
  "inversion": {
    "algorithm": 6,
    "max_eval": 200,
    "initial_grain_radius_um": 250,
    "apply_valid_inversion_mask": true,
    "n_workers": 1
  }
}
```

`max_eval` may be omitted to use the inversion engine's algorithm-specific
default: 200 for Algorithm 6 and 100 for Algorithms 1–5. The Algorithm 6 grain
step is intentionally owned by `spires-inversion`; it is not a run-config
option and is recorded in inversion-result provenance.

Configuration parsing does not execute inversion. Until the top-level
`spires` workflow wires the complete pipeline, pass the validated settings
explicitly:

```python
from spires_io.configs import SpiresConfig
import spires_inversion
import spires_io

config = SpiresConfig("config.json")
data = spires_io.load("config.json")
inverted = spires_inversion.invert(
    data,
    lut=config.files.lut,
    **config.inversion.to_invert_kwargs(),
)
```

## Postprocessing inputs

`spires-io` loads and aligns the static inputs needed by
`spires-postprocess`, but it does not apply canopy or glacier-ice science
adjustments. A single-scene configuration can opt into those later operations:

```json
{
  "files": {
    "image_data": "scene.hdf",
    "background_image": "background.nc",
    "lut": "lut.mat",
    "canopy_fraction": "canopy_fraction.tif",
    "ice_fraction": "ice_fraction.tif",
    "slope": "slope.tif",
    "aspect": "aspect.tif"
  },
  "sensor": {"name": "viirs"},
  "postprocess": {
    "apply_canopy_correction": true,
    "apply_ice_adjustment": true,
    "calculate_albedo": true,
    "calculate_delta_vis": false,
    "calculate_radiative_forcing": false,
    "average_vertical_crown_radius": 4.644,
    "average_horizontal_crown_radius": 1.72
  }
}
```

Canopy and ice rasters may store either fractions in `[0, 1]` or integer
percentages in `[0, 100]`. Percentage products are divided by 100 and exposed as
`float32` ancillary variables named `canopy_fraction` and `ice_fraction`.

`ice_fraction` is a numeric scientific input and never changes
`valid_inversion_mask`. The separate `files.ice_mask` setting remains available
only for an explicitly requested binary inversion-exclusion mask. In particular,
an RGI fractional-ice raster should be configured as `files.ice_fraction`, not
`files.ice_mask`.

Batch manifests and run-wide policy belong to `spires-batch`. Batch execution
calls the public scene, background, ancillary, mask, and persistence APIs
directly; `spires-io` does not define a second manifest-item schema.

## Raster scene adapters (Landsat, HLS, Sentinel-2)

`spires_io.raster_scene.open_raster_scene` turns a locally stored scene directory
into an inversion-ready `xarray.Dataset` on one grid:

| Variable       | Dims            | dtype   | Meaning                                                    |
|----------------|-----------------|---------|------------------------------------------------------------|
| `reflectance`  | `(y, x, band)`  | float32 | Surface reflectance; NaN where nodata. Passes `validate_target_spectra`. |
| `valid`        | `(y, x)`        | bool    | True = run the inversion (not fill / cloud / shadow / cirrus / saturated / nodata). |
| `solar_zenith` | `(y, x)`        | float32 | Degrees, `units="degree"`. Passes `validate_solar_angles`. |

`band` carries common names (`blue, green, red, nir, swir1, swir2, …`) and a
`wavelength_nm` coordinate. Dataset attrs: `sensor`, `platform` (Landsat), `acquisition_time`,
`resolution_m`, `source_product_id`; the CRS is carried by the `rio` accessor (`ds.rio.crs`).

Each sensor is a `spires_io.sensor_spec.SensorSpec` (band globs, scale/offset/nodata,
QA decoder, acquisition time, solar-zenith provider). Registered adapters:

| Registry key | Scene directory layout                                          | Solar zenith                      |
|--------------|-----------------------------------------------------------------|-----------------------------------|
| `landsat`    | Collection 2 Level-2 bundle: `*_SR_B2..B7.TIF`, `*_QA_PIXEL.TIF`, `*_QA_RADSAT.TIF`, `*_MTL.txt` | computed per pixel (`spires_io.solar`) |
| `hls_l30`    | `HLS.L30.*.v2.0.B01..B07.tif`, `.Fmask.tif`, `.SZA.tif`           | shipped `SZA` band                |
| `hls_s30`    | `HLS.S30.*.v2.0.B01..B12.tif`, `.Fmask.tif`, `.SZA.tif`           | shipped `SZA` band                |
| `sentinel2`  | Element84 Earth Search COGs: `B02.tif … B12.tif`, `SCL.tif`, plus the STAC item as `item.json` | computed per pixel (`spires_io.solar`) |

```python
from spires_io.api import open_surface_reflectance

ds = open_surface_reflectance("/scratch/spires/scenes/landsat/LC08_L2SP_042034_20240214_20240223_02_T1", sensor="landsat")
s2 = open_surface_reflectance(".../sentinel2/S2B_11SLB_20250420_0_L2A", sensor="sentinel2")   # 20 m, 10 m bands block-averaged
s2_10m = open_surface_reflectance(".../sentinel2/S2B_11SLB_20250420_0_L2A", sensor="sentinel2",
                                  target_resolution_m=10, resampler=my_super_resolver)       # any Resampler callable
```

A `Resampler` is `Callable[[dict[str, DataArray], int], dict[str, DataArray]]`: all
native-resolution bands in, all bands on the target grid out. `block_mean` (default when
coarsening) and `nearest` ship with `spires_io.sensor_spec`; a super-resolver such as DSen2
is supplied by the caller and never imported here.

Scene root convention: `/scratch/spires/scenes/<sensor>/<scene_id>/`. Adapters never
touch the network; `spires-fetch` (optional extra `pip install -e ".[fetch]"`) populates it:

```bash
spires-fetch --dest /scratch/spires/scenes --aoi mammoth --start 2026-10-01 --end 2026-11-06 --max-scenes 1
spires-fetch --dest /scratch/spires/scenes --scene-id HLS.L30.T11SLB.2026289T182301.v2.0   # exact ids, no search
spires-fetch --dest /scratch/spires/scenes --aoi mammoth --dry-run                           # show the selection only
```

Sources: Landsat C2 L2 from Planetary Computer (anonymous SAS token), HLS v2.0 from
CMR-STAC LPCLOUD (needs a `urs.earthdata.nasa.gov` entry in `~/.netrc`), Sentinel-2 L2A
from Element84 Earth Search (`sentinel-2-l2a`, whose COGs are already BOA-offset-harmonized;
the adapter refuses items without `earthsearch:boa_offset_applied: true` rather than
guessing the offset). Superseded reprocessings are dropped per satellite pass and one scene
is kept per date; scenes download into `<scene_id>.partial/` and are renamed into place only
when complete. Original Copernicus SAFE products are not supported.

Opt-in real-data tests: `SPIRES_SCENE_CACHE=/scratch/spires/scenes pytest -m real_data`.

## Solar and terrain illumination geometry

Prepared scenes always include `cosine_solar_zenith` when `solar_zenith` is
available. When scene solar azimuth and aligned `slope` and `aspect` ancillary
layers are also available, they include `cosine_illumination`, the cosine of
local solar incidence:

```text
mu_i = sin(zenith) sin(slope) cos(solar_azimuth - aspect)
       + cos(zenith) cos(slope)
cosine_illumination = clip(mu_i, 0, 1)
```

Source angles are degrees. Solar azimuth and terrain aspect use degrees
clockwise from north in `[0, 360)`; finite solar azimuth is normalized modulo
360. Invalid or missing geometry remains `NaN`. A flat surface therefore has
`cosine_illumination == cosine_solar_zenith` within floating-point tolerance.
The quantity describes local surface incidence only and does not model shadows
cast by surrounding terrain.

The three `calculate_*` postprocess flags are independent workflow-intent
settings and default to `false`. They do not dispatch postprocessing in this
phase. `calculate_albedo: true` does require enough geometry to derive
`cosine_illumination`; missing solar azimuth, slope, or aspect raises a clear
loading error.

## Optional illumination clustering

`cosine_illumination` is an opt-in clustering feature alongside reflectance,
background reflectance, and solar zenith. Its default dimensionless tolerance
is `0.02` and can be tuned with `clustering.cosine_illumination_tol`. Any
nonempty feature subset is supported, including scalar-only selections:

```json
{
  "clustering": {
    "enabled": true,
    "features": ["solar_zenith", "cosine_illumination"],
    "solar_zenith_tol": 2.0,
    "cosine_illumination_tol": 0.02
  }
}
```

Only selected features determine grouping. Reflectance, background, and solar
zenith remain required inversion payloads: their finite values determine basic
eligibility and their representatives are always emitted. The clustered scene
includes `cluster_representative_cosine_illumination` when that feature is
selected. Configured loading requires derivable illumination geometry only
when clustering is enabled and the feature is selected.

Clustering is explicit and does not run during loading:

```python
from spires_io import cluster

clustered = cluster(
    data,
    **config.clustering.to_cluster_kwargs(),
    apply_valid_inversion_mask=config.inversion.apply_valid_inversion_mask,
)
```

The mask policy is recorded as `valid_inversion_mask_applied` on
`cluster_label`. The corresponding configuration option is
`inversion.apply_valid_inversion_mask`, which defaults to `true`.

Clustering currently materializes scene arrays in memory. Pass the returned
object to `spires_inversion.invert()` for automatic representative inversion
and spatial scattering. Postprocessing continues to evaluate pixel-level
illumination after inversion products have been scattered.
