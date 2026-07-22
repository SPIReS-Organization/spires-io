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

Complete in-memory objects can be written atomically to a grouped NetCDF
product and reconstructed for a later scientific stage:

```python
output_path = spires_io.write_spires_data(data, "prepared_scene.nc")
restored = spires_io.read_spires_data(output_path)
```

The initial layout stores the required `scene` field and any populated
`background`, `ancillary`, and `results` fields in matching NetCDF groups.
Existing destinations are preserved unless overwrite is explicitly enabled.

## Inversion-exclusion provenance

Prepared MODIS and VIIRS scenes store inversion exclusions in the canonical
`uint16` pair `inversion_exclusion_flags` and
`inversion_exclusion_assessed`, together with `valid_inversion_mask`. Use
`decode_inversion_exclusions()` to regenerate Boolean reason and assessed-state
masks. `write_detailed_masks` has been removed because the packed pair is the
lossless, canonical representation.

For VIIRS, all QF1-QF7 bytes are read even when reflectance is down-selected.
QF3-QF6 band-specific SDR-input and surface-reflectance-quality flags are
decoded for every supported reflective band on the separate `qa_band`
coordinate. A pixel receives the `poor_surface_reflectance_quality` exclusion
when any selected band is flagged, or when a required atmospheric-correction
input is bad or missing. Flags belonging only to unselected bands remain
available as diagnostics but do not exclude the pixel.

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

For scene manifests, the corresponding paths belong in each scene's
`ancillary` mapping. When a postprocessing operation is enabled, its required
ancillary path is validated as the scene is dispatched.

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

Clustering currently materializes scene arrays in memory. Cluster-to-inversion
handoff and scattering are deferred; postprocessing continues to evaluate
pixel-level illumination after inversion products have been scattered.
