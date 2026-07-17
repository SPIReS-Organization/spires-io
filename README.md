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
