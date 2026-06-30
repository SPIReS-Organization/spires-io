import json
import warnings

import pytest

from spires_io.configs import SpiresConfig


def _write_config(tmp_path, files):
    config_path = tmp_path / "spires_config.json"
    config_path.write_text(
        json.dumps(
            {
                "files": files,
                "sensor": {"name": "modis"},
            }
        )
    )
    return config_path


def _required_files(**overrides):
    files = {
        "image_data": "scene.hdf",
        "background_image": "background.tif",
        "lut": "lut.mat",
    }
    files.update(overrides)
    return files


def test_spires_config_accepts_background_image_without_warning(tmp_path):
    config_path = _write_config(tmp_path, _required_files())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = SpiresConfig(config_path)

    assert caught == []
    assert config.files.background_image == "background.tif"
    assert config.files.snowfree_image == "background.tif"


def test_spires_config_accepts_tiff_background_image(tmp_path):
    background = tmp_path / "background.tiff"
    background.touch()
    config_path = _write_config(
        tmp_path,
        _required_files(background_image=str(background)),
    )

    config = SpiresConfig(config_path)

    assert config.files.background_image == str(background)


def test_spires_config_accepts_deprecated_snowfree_image_with_warning(tmp_path):
    files = _required_files(snowfree_image="snowfree.tif")
    del files["background_image"]
    config_path = _write_config(tmp_path, files)

    with pytest.warns(FutureWarning, match="snowfree_image is deprecated"):
        config = SpiresConfig(config_path)

    assert config.files.background_image == "snowfree.tif"
    assert config.files.snowfree_image == "snowfree.tif"


def test_spires_config_rejects_background_and_snowfree_image_together(tmp_path):
    config_path = _write_config(
        tmp_path,
        _required_files(snowfree_image="snowfree.tif"),
    )

    with pytest.raises(ValueError, match="Use only one"):
        SpiresConfig(config_path)


def test_spires_config_requires_background_image_or_deprecated_alias(tmp_path):
    files = _required_files()
    del files["background_image"]
    config_path = _write_config(tmp_path, files)

    with pytest.raises(ValueError, match="background_image"):
        SpiresConfig(config_path)


def test_spires_config_rejects_moved_options(tmp_path):
    config_path = tmp_path / "spires_config.json"
    config_path.write_text(
        json.dumps(
            {
                "files": _required_files(),
                "sensor": {"name": "modis"},
                "options": {"max_sensor_zenith": 50.0},
            }
        )
    )

    with pytest.raises(ValueError, match="options.max_sensor_zenith -> reader"):
        SpiresConfig(config_path)
