# tests/conftest.py
"""Synthetic Landsat / HLS / Sentinel-2 scene directories for offline tests."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

CRS = "EPSG:32611"
X0, Y0 = 300000.0, 4170000.0   # upper-left corner, UTM 11N, near Mammoth
SHAPE = (8, 8)


def write_raster(path: Path, values: np.ndarray, resolution_m: int, nodata=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype=values.dtype, crs=CRS, nodata=nodata,
        transform=Affine(resolution_m, 0.0, X0, 0.0, -resolution_m, Y0),
    ) as dst:
        dst.write(values, 1)
    return path


def _fill(dn, name, shape, dtype):
    """``dn`` is a scalar for all bands or a dict band -> scalar | 2-D array (default 4000)."""
    value = dn.get(name, 4000) if isinstance(dn, dict) else dn
    return np.full(shape, value, dtype=dtype) if np.ndim(value) == 0 else np.asarray(value, dtype=dtype)


LANDSAT_BANDS = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6", "swir2": "B7"}
LANDSAT_ID = "LC08_L2SP_042034_20240214_20240223_02_T1"
# bit 6 (clear) set; cloud / shadow / snow / cirrus confidence all "low" (01); no fill/cloud bits.
LANDSAT_CLEAR_QA_PIXEL = 21824


def write_fake_landsat_scene(root: Path, *, dn=4000, qa=None, shape=SHAPE) -> Path:
    scene = root / LANDSAT_ID
    for name, code in LANDSAT_BANDS.items():
        write_raster(scene / f"{LANDSAT_ID}_SR_{code}.TIF", _fill(dn, name, shape, np.uint16), 30, nodata=0)
    qa = qa or {}
    qa_pixel = np.asarray(qa.get("qa_pixel", np.full(shape, LANDSAT_CLEAR_QA_PIXEL)), dtype=np.uint16)
    qa_radsat = np.asarray(qa.get("qa_radsat", np.zeros(shape)), dtype=np.uint16)
    write_raster(scene / f"{LANDSAT_ID}_QA_PIXEL.TIF", qa_pixel, 30)
    write_raster(scene / f"{LANDSAT_ID}_QA_RADSAT.TIF", qa_radsat, 30)
    (scene / f"{LANDSAT_ID}_MTL.txt").write_text(
        "GROUP = LANDSAT_METADATA_FILE\n"
        "  GROUP = IMAGE_ATTRIBUTES\n"
        "    SPACECRAFT_ID = \"LANDSAT_8\"\n"
        "    DATE_ACQUIRED = 2024-02-14\n"
        "    SCENE_CENTER_TIME = \"18:23:01.1234567Z\"\n"
        "  END_GROUP = IMAGE_ATTRIBUTES\n"
        "END_GROUP = LANDSAT_METADATA_FILE\n"
    )
    return scene


HLS_L30_BANDS = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "nir": "B05", "swir1": "B06", "swir2": "B07"}
HLS_S30_BANDS = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "red_edge1": "B05",
                 "red_edge2": "B06", "red_edge3": "B07", "nir": "B08", "nir_narrow": "B8A",
                 "swir1": "B11", "swir2": "B12"}


def write_fake_hls_scene(root: Path, product="L30", *, dn=4000, qa=None, sza=45.0, shape=SHAPE) -> Path:
    bands = HLS_L30_BANDS if product == "L30" else HLS_S30_BANDS
    scene_id = f"HLS.{product}.T11SLB.2024045T182301.v2.0"
    scene = root / scene_id
    for name, code in bands.items():
        path = write_raster(scene / f"{scene_id}.{code}.tif", _fill(dn, name, shape, np.int16), 30, nodata=-9999)
        with rasterio.open(path, "r+") as dst:
            dst.update_tags(SENSING_TIME="2024-02-14T18:23:01.1234567Z")
    fmask = (qa or {}).get("fmask", np.zeros(shape, np.uint8))
    write_raster(scene / f"{scene_id}.Fmask.tif", np.asarray(fmask, dtype=np.uint8), 30, nodata=255)
    write_raster(scene / f"{scene_id}.SZA.tif", np.asarray(np.full(shape, sza) * 100.0, dtype=np.uint16), 30)
    return scene


S2_10M = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"}
S2_20M = {"red_edge1": "B05", "red_edge2": "B06", "red_edge3": "B07", "nir_narrow": "B8A", "swir1": "B11", "swir2": "B12"}
S2_ID = "S2B_11SLB_20250420_0_L2A"   # Element84 Earth Search item id


def write_fake_sentinel2_scene(root: Path, *, dn=4000, qa=None, boa_offset_applied=True,
                               shape_10m=(16, 16), shape_20m=SHAPE) -> Path:
    """Element84 COG layout: flat B??.tif + SCL.tif + the STAC item saved as item.json."""
    scene = root / S2_ID
    for name, code in S2_10M.items():
        write_raster(scene / f"{code}.tif", _fill(dn, name, shape_10m, np.uint16), 10, nodata=0)
    for name, code in S2_20M.items():
        write_raster(scene / f"{code}.tif", _fill(dn, name, shape_20m, np.uint16), 20, nodata=0)
    scl = (qa or {}).get("scl", np.full(shape_20m, 4, np.uint8))
    write_raster(scene / "SCL.tif", np.asarray(scl, dtype=np.uint8), 20, nodata=0)
    item = {
        "type": "Feature", "stac_version": "1.1.0", "id": S2_ID, "collection": "sentinel-2-l2a",
        "properties": {
            "datetime": "2025-04-20T18:54:03.375000Z",
            "platform": "sentinel-2b",
            "grid:code": "MGRS-11SLB",
            "s2:processing_baseline": "05.11",
            "earthsearch:boa_offset_applied": boa_offset_applied,
        },
    }
    (scene / "item.json").write_text(json.dumps(item, indent=2))
    return scene


@pytest.fixture
def landsat_scene(tmp_path):
    return write_fake_landsat_scene(tmp_path)


@pytest.fixture
def hls_l30_scene(tmp_path):
    return write_fake_hls_scene(tmp_path, "L30")


@pytest.fixture
def hls_s30_scene(tmp_path):
    return write_fake_hls_scene(tmp_path, "S30")


@pytest.fixture
def sentinel2_scene(tmp_path):
    return write_fake_sentinel2_scene(tmp_path)
