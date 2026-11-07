# tests/fetch_helpers.py
"""Hand-built pystac Items shaped like the three catalogs, and a local HTTP server for download tests."""
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pystac

MAMMOTH_BBOX = (-119.049, 37.623, -119.009, 37.663)
TILE_BBOX = (-119.3, 37.3, -118.3, 38.2)


def _poly(bbox):
    w, s, e, n = bbox
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def make_stac_item(item_id, when, properties, assets, bbox=TILE_BBOX, collection=None):
    """``when`` is an ISO-8601 string; ``assets`` maps asset key -> href (plus optional ``file:size`` via a tuple)."""
    item = pystac.Item(
        id=item_id, geometry=_poly(bbox), bbox=list(bbox),
        datetime=pystac.utils.str_to_datetime(when), properties=dict(properties), collection=collection,
    )
    for key, value in assets.items():
        href, size = value if isinstance(value, tuple) else (value, None)
        asset = pystac.Asset(href=href)
        if size is not None:
            asset.extra_fields["file:size"] = size
        item.add_asset(key, asset)
    return item


# ---- Planetary Computer landsat-c2-l2 ---------------------------------------
LANDSAT_SUFFIX = {
    "blue": "SR_B2.TIF", "green": "SR_B3.TIF", "red": "SR_B4.TIF", "nir08": "SR_B5.TIF",
    "swir16": "SR_B6.TIF", "swir22": "SR_B7.TIF", "qa_pixel": "QA_PIXEL.TIF",
    "qa_radsat": "QA_RADSAT.TIF", "mtl.txt": "MTL.txt",
}


def landsat_item(item_id="LC08_L2SP_042034_20240214_02_T1", when="2024-02-14T18:23:01Z",
                 created="2024-02-23T00:00:00Z", platform="landsat-8", path="042", row="034",
                 category="T1", cloud=5.0, bbox=TILE_BBOX):
    usgs_id = item_id.replace("_02_T1", "_20240223_02_T1")   # PC ids omit the processing date; filenames keep it
    base = ("https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/standard/oli-tirs/"
            f"2024/{path}/{row}/{usgs_id}")
    props = {"platform": platform, "created": created, "landsat:wrs_path": path, "landsat:wrs_row": row,
             "landsat:collection_category": category, "eo:cloud_cover": cloud}
    assets = {key: f"{base}/{usgs_id}_{suffix}" for key, suffix in LANDSAT_SUFFIX.items()}
    return make_stac_item(item_id, when, props, assets, bbox=bbox, collection="landsat-c2-l2")


# ---- CMR-STAC LPCLOUD HLSL30_2.0 / HLSS30_2.0 ---------------------------------
HLS_L30_CODES = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "Fmask", "SZA")
HLS_S30_CODES = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "Fmask", "SZA")


def hls_item(product="L30", tile="T11SLB", stamp="2024045T182301", when="2024-02-14T18:23:01Z",
             created=None, cloud=5.0, bbox=TILE_BBOX):
    item_id = f"HLS.{product}.{tile}.{stamp}.v2.0"
    base = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLS{product}.020/{item_id}"
    codes = HLS_L30_CODES if product == "L30" else HLS_S30_CODES
    props = {"eo:cloud_cover": cloud, "start_datetime": when, "end_datetime": when}
    if created:
        props["created"] = created
    assets = {code: f"{base}/{item_id}.{code}.tif" for code in codes}
    return make_stac_item(item_id, when, props, assets, bbox=bbox, collection=f"HLS{product}_2.0")


# ---- Earth Search v1 sentinel-2-l2a ---------------------------------------------
S2_CODES = {
    "blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06", "rededge3": "B07",
    "nir": "B08", "nir08": "B8A", "swir16": "B11", "swir22": "B12", "scl": "SCL",
}


def s2_item(item_id="S2B_11SLB_20250420_0_L2A", when="2025-04-20T18:54:03.375000Z",
            datatake="GS2B_20250420T183909_042410_N05.11", baseline="05.11", created="2025-04-21T00:00:00Z",
            platform="sentinel-2b", boa=True, nodata=0.0, cloud=5.0, tile="MGRS-11SLB", bbox=TILE_BBOX):
    base = f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/11/S/LB/2025/4/{item_id}"
    props = {"platform": platform, "created": created, "updated": created, "grid:code": tile,
             "s2:datatake_id": datatake, "s2:processing_baseline": baseline,
             "s2:nodata_pixel_percentage": nodata, "eo:cloud_cover": cloud}
    if boa is not None:
        props["earthsearch:boa_offset_applied"] = boa
    assets = {key: f"{base}/{code}.tif" for key, code in S2_CODES.items()}
    return make_stac_item(item_id, when, props, assets, bbox=bbox, collection="sentinel-2-l2a")


# ---- local HTTP server -----------------------------------------------------------
class FakeAssetServer:
    """Serves ``files`` (URL path -> bytes). ``fail_first[path] = n`` answers 503 to the first n GETs of that path.
    ``truncate_first[path] = n`` sends only the first n body bytes of the first GET (Content-Length stays full), then closes."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.fail_first: dict[str, int] = {}
        self.truncate_first: dict[str, int] = {}
        self.hits: dict[str, int] = defaultdict(int)
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.hits[self.path] += 1
                if server.fail_first.get(self.path, 0) >= server.hits[self.path]:
                    self.send_error(503)
                    return
                body = server.files.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                cut = server.truncate_first.get(self.path)
                if cut is not None and server.hits[self.path] == 1:
                    self.wfile.write(body[:cut])
                    return
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
