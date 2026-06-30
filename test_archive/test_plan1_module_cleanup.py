import json


def test_config_normalizes_sensor_alias_and_exposes_band_metadata(tmp_path):
    from spires_io.configs import SpiresConfig

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "files": {
                    "image_data": "scene.hdf",
                    "background_image": "background.nc",
                    "lut": "lut.mat",
                },
                "sensor": {"name": "terra", "selected_bands": ["1", "2"]},
            }
        )
    )

    config = SpiresConfig(config_path)

    assert config.sensor.name == "modis"
    assert config.sensor.band_names.tolist() == ["1", "2"]
    assert config.sensor.wavelength.tolist() == [645.0, 858.5]


def test_file_type_constants_are_shared():
    from spires_io.file_types import ALL_SUPPORTED_FILE_SUFFIXES, RASTER_SUFFIXES

    assert ".hdf" in ALL_SUPPORTED_FILE_SUFFIXES
    assert ".zarr" in ALL_SUPPORTED_FILE_SUFFIXES
    assert RASTER_SUFFIXES == frozenset({".tif", ".tiff"})
