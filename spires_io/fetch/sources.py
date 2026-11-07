# spires_io/fetch/sources.py
"""Per-sensor STAC source configuration for ``spires-fetch``.

Nothing here imports the adapters: which assets to download is a catalog-side
fact. ``tests/test_fetch_sources.py`` cross-checks the asset lists against the
adapter globs (over hand-written hrefs mirroring each catalog), so a change on
the adapter side fails offline; a change in the catalogs' own naming is only
caught by the ``real_data`` acceptance run.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

import pystac
import requests

Bbox = tuple[float, float, float, float]   # W, S, E, N in degrees

# Jeff Dozier Snow Study Site (formerly CUES), 37.643 N 119.029 W, +-0.02 deg; inside T11SLB / WRS-2 042034.
AOIS: dict[str, Bbox] = {"mammoth": (-119.049, 37.623, -119.009, 37.663)}

PC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
CMR_LPCLOUD_ENDPOINT = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
EARTH_SEARCH_ENDPOINT = "https://earth-search.aws.element84.com/v1"


@dataclass(frozen=True)
class Source:
    sensor: str
    endpoint: str
    collection: str
    assets: tuple[str, ...]                                  # STAC asset keys to download, in order
    acquisition_key: Callable[[pystac.Item], tuple]          # same satellite pass -> same key
    tile: Callable[[pystac.Item], str]
    keep: Callable[[pystac.Item], bool]                      # pre-filter
    nodata_pct: Callable[[pystac.Item], float | None]        # catalog's nodata fraction, None if absent
    needs_earthdata: bool
    make_signer: Callable[[], Callable[[str], str]]          # called once per run; returns href -> fetchable URL


def item_datetime(item: pystac.Item) -> dt.datetime:
    if item.datetime is not None:
        return item.datetime
    return pystac.utils.str_to_datetime(item.properties["start_datetime"])


def created_key(item: pystac.Item) -> tuple:
    """Sort key for "most recently processed": created/updated, then processing baseline, then id."""
    stamp = item.properties.get("created") or item.properties.get("updated")
    created = pystac.utils.str_to_datetime(stamp) if stamp else dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    return (created, item.properties.get("s2:processing_baseline", ""), item.id)


def asset_filename(href: str) -> str:
    """Original product filename: the last path segment of the unsigned href."""
    return urlsplit(href).path.rsplit("/", 1)[-1]


def identity_signer() -> Callable[[str], str]:
    return lambda href: href


def planetary_computer_signer(session: requests.Session | None = None) -> Callable[[str], str]:
    """Fetch one anonymous SAS token for ``landsat-c2-l2`` and return a function appending it to blob hrefs."""
    session = session or requests.Session()
    response = session.get(PC_SAS_URL, timeout=30)
    response.raise_for_status()
    token = response.json()["token"]
    return lambda href: f"{href}?{token}"


def _landsat_acquisition(item):
    return (item_datetime(item).date(), item.properties["platform"])


def _landsat_tile(item):
    return f'{item.properties["landsat:wrs_path"]}{item.properties["landsat:wrs_row"]}'


def _hls_acquisition(item):
    return (item.id.split(".")[3],)          # HLS.L30.T11SLB.2024045T182301.v2.0 -> sensing stamp (to the second)


def _hls_tile(item):
    return item.id.split(".")[2]


def _s2_acquisition(item):
    return (item.properties["s2:datatake_id"].rsplit("_N", 1)[0],)   # strip processing baseline


def _s2_tile(item):
    return item.properties["grid:code"]


def _s2_nodata(item):
    value = item.properties.get("s2:nodata_pixel_percentage")
    return None if value is None else float(value)


def _hls_source(sensor: str, collection: str, assets: tuple[str, ...]) -> Source:
    return Source(
        sensor=sensor, endpoint=CMR_LPCLOUD_ENDPOINT, collection=collection, assets=assets,
        acquisition_key=_hls_acquisition, tile=_hls_tile, keep=lambda item: True,
        nodata_pct=lambda item: None, needs_earthdata=True, make_signer=identity_signer,
    )


SOURCES: dict[str, Source] = {
    "landsat": Source(
        sensor="landsat", endpoint=PC_ENDPOINT, collection="landsat-c2-l2",
        assets=("blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel", "qa_radsat", "mtl.txt"),
        acquisition_key=_landsat_acquisition, tile=_landsat_tile,
        keep=lambda item: item.properties.get("landsat:collection_category") == "T1",
        nodata_pct=lambda item: None, needs_earthdata=False, make_signer=planetary_computer_signer,
    ),
    "hls_l30": _hls_source("hls_l30", "HLSL30_2.0",
                           ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "Fmask", "SZA")),
    "hls_s30": _hls_source("hls_s30", "HLSS30_2.0",
                           ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "Fmask", "SZA")),
    "sentinel2": Source(
        sensor="sentinel2", endpoint=EARTH_SEARCH_ENDPOINT, collection="sentinel-2-l2a",
        assets=("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22", "scl"),
        acquisition_key=_s2_acquisition, tile=_s2_tile,
        keep=lambda item: item.properties.get("earthsearch:boa_offset_applied") is True,
        nodata_pct=_s2_nodata, needs_earthdata=False, make_signer=identity_signer,
    ),
}


def sensor_from_scene_id(scene_id: str) -> str:
    if scene_id.startswith(("LC08_", "LC09_")):
        return "landsat"
    if scene_id.startswith("HLS.L30."):
        return "hls_l30"
    if scene_id.startswith("HLS.S30."):
        return "hls_s30"
    if scene_id.startswith(("S2A_", "S2B_", "S2C_")):
        return "sentinel2"
    raise ValueError(f"cannot infer sensor from scene id {scene_id!r}")
