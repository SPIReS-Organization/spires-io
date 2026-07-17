# spires-io

I/O for the [SPIReS](https://github.com/SPIReS-Organization) package family:
MODIS / Sentinel-2 / Landsat loaders, reprojection, and coordinate transforms.

Produces target/background reflectance spectra and solar angles that conform to
the I/O→inversion boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract).

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

Only selected features determine grouping and finite-value eligibility. The
clustered scene includes `cluster_representative_cosine_illumination` when that
feature is selected. Configured loading requires derivable illumination
geometry only when clustering is enabled and the feature is selected; a direct
`SpiresData.cluster()` request raises `ValueError` if it is absent.

Clustering currently materializes scene arrays in memory. Cluster-to-inversion
handoff and scattering are deferred; postprocessing continues to evaluate
pixel-level illumination after inversion products have been scattered.
