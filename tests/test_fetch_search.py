# tests/test_fetch_search.py
import pytest

pytest.importorskip("pystac")
from fetch_helpers import MAMMOTH_BBOX, hls_item, landsat_item, s2_item

from spires_io.fetch import search
from spires_io.fetch.sources import SOURCES

S2 = SOURCES["sentinel2"]
LANDSAT = SOURCES["landsat"]
HLS = SOURCES["hls_l30"]


def ids(items):
    return sorted(item.id for item in items)


def test_prefilter_drops_s2_without_boa_offset():
    items = [s2_item(item_id="A", boa=True), s2_item(item_id="B", boa=False), s2_item(item_id="C", boa=None)]
    assert ids(search.prefilter(S2, items)) == ["A"]


def test_prefilter_drops_landsat_t2():
    items = [landsat_item(item_id="LC08_T1", category="T1"), landsat_item(item_id="LC08_T2", category="T2")]
    assert ids(search.prefilter(LANDSAT, items)) == ["LC08_T1"]


def test_latest_per_acquisition_keeps_newest_baseline():
    old = s2_item(item_id="S2B_11SLB_20200804_0_L2A", datatake="GS2B_20200804T183919_017828_N02.14",
                  baseline="02.14", created="2022-11-06T00:00:00Z", when="2020-08-04T18:39:19Z")
    new = s2_item(item_id="S2B_11SLB_20200804_1_L2A", datatake="GS2B_20200804T183919_017828_N05.00",
                  baseline="05.00", created="2023-09-14T00:00:00Z", when="2020-08-04T18:39:19Z")
    assert ids(search.latest_per_acquisition(S2, [old, new])) == ["S2B_11SLB_20200804_1_L2A"]
    assert ids(search.latest_per_acquisition(S2, [new, old])) == ["S2B_11SLB_20200804_1_L2A"]


def test_latest_per_acquisition_tie_breaks_on_baseline():
    low = s2_item(item_id="S2B_11SLB_20200804_0_L2A", datatake="GS2B_20200804T183919_017828_N02.14",
                  baseline="02.14", created="2023-09-14T00:00:00Z", when="2020-08-04T18:39:19Z")
    high = s2_item(item_id="S2B_11SLB_20200804_1_L2A", datatake="GS2B_20200804T183919_017828_N05.00",
                   baseline="05.00", created="2023-09-14T00:00:00Z", when="2020-08-04T18:39:19Z")
    assert ids(search.latest_per_acquisition(S2, [low, high])) == ["S2B_11SLB_20200804_1_L2A"]
    assert ids(search.latest_per_acquisition(S2, [high, low])) == ["S2B_11SLB_20200804_1_L2A"]


def test_latest_per_acquisition_keeps_two_overpasses_same_day():
    a = s2_item(item_id="S2A_11SLB_20250507_0_L2A", datatake="GS2A_20250507T184821_051576_N05.11",
                platform="sentinel-2a", when="2025-05-07T18:54:13Z")
    b = s2_item(item_id="S2B_11SLB_20250507_0_L2A", datatake="GS2B_20250507T182919_042667_N05.11",
                platform="sentinel-2b", when="2025-05-07T18:44:06Z")
    assert ids(search.latest_per_acquisition(S2, [a, b])) == [a.id, b.id]


def test_latest_per_acquisition_keeps_different_tiles_of_one_pass():
    a = s2_item(item_id="S2B_11SLB", tile="MGRS-11SLB")
    b = s2_item(item_id="S2B_11SLA", tile="MGRS-11SLA")
    assert ids(search.latest_per_acquisition(S2, [b, a])) == ["S2B_11SLA", "S2B_11SLB"]


def test_latest_per_acquisition_hls_dedupes_identical_ids():
    a = hls_item("L30", created="2024-02-20T00:00:00Z")
    b = hls_item("L30", created="2024-03-01T00:00:00Z")
    for order in ([b, a], [a, b]):
        kept = search.latest_per_acquisition(HLS, order)
        assert len(kept) == 1 and kept[0].properties["created"] == "2024-03-01T00:00:00Z"


def test_one_per_date_prefers_lowest_nodata():
    full = s2_item(item_id="S2A_full", datatake="GS2A_20250507T184821_051576_N05.11", platform="sentinel-2a",
                   when="2025-05-07T18:54:13Z", nodata=0.0)
    part = s2_item(item_id="S2B_part", datatake="GS2B_20250507T182919_042667_N05.11", platform="sentinel-2b",
                   when="2025-05-07T18:44:06Z", nodata=52.7)
    assert ids(search.one_per_date(S2, [part, full], MAMMOTH_BBOX)) == ["S2A_full"]


def test_one_per_date_ranks_missing_nodata_last():
    with_nodata = s2_item(item_id="S2A_with", datatake="GS2A_20250507T184821_051576_N05.11",
                          platform="sentinel-2a", when="2025-05-07T18:54:13Z", nodata=30.0)
    without = s2_item(item_id="S2B_without", datatake="GS2B_20250507T182919_042667_N05.11",
                      platform="sentinel-2b", when="2025-05-07T18:44:06Z")
    del without.properties["s2:nodata_pixel_percentage"]
    assert ids(search.one_per_date(S2, [without, with_nodata], MAMMOTH_BBOX)) == ["S2A_with"]
    assert ids(search.one_per_date(S2, [with_nodata, without], MAMMOTH_BBOX)) == ["S2A_with"]


def test_one_per_date_falls_back_to_bbox_overlap():
    inside = landsat_item(item_id="LC08_042034", row="034", bbox=(-120.0, 37.0, -118.0, 38.5))
    edge = landsat_item(item_id="LC08_042035", row="035", bbox=(-120.0, 36.0, -118.0, 37.64))
    assert ids(search.one_per_date(LANDSAT, [inside, edge], MAMMOTH_BBOX)) == ["LC08_042034"]


def test_one_per_date_keeps_different_dates():
    a = landsat_item(item_id="LC08_A", when="2024-02-14T18:23:01Z")
    b = landsat_item(item_id="LC09_B", when="2024-02-22T18:23:01Z", platform="landsat-9")
    assert ids(search.one_per_date(LANDSAT, [b, a], MAMMOTH_BBOX)) == ["LC08_A", "LC09_B"]


def test_select_applies_cloud_filter_sort_and_cap():
    items = [
        landsat_item(item_id="LC08_C", when="2024-03-01T18:23:01Z", cloud=90.0),
        landsat_item(item_id="LC08_A", when="2024-02-14T18:23:01Z", cloud=5.0),
        landsat_item(item_id="LC09_B", when="2024-02-22T18:23:01Z", platform="landsat-9", cloud=20.0),
        landsat_item(item_id="LC08_T2", when="2024-02-06T18:23:01Z", category="T2"),
    ]
    chosen = search.select(LANDSAT, items, MAMMOTH_BBOX, max_cloud=50.0)
    assert [item.id for item in chosen] == ["LC08_A", "LC09_B"]
    assert [item.id for item in search.select(LANDSAT, items, MAMMOTH_BBOX, max_scenes=1)] == ["LC08_A"]


def test_select_drops_items_without_cloud_cover_when_filtering():
    known = landsat_item(item_id="LC08_known", when="2024-02-14T18:23:01Z", cloud=5.0)
    unknown = landsat_item(item_id="LC08_unknown", when="2024-02-22T18:23:01Z", cloud=5.0)
    del unknown.properties["eo:cloud_cover"]
    assert ids(search.select(LANDSAT, [known, unknown], MAMMOTH_BBOX, max_cloud=50.0)) == ["LC08_known"]
    assert ids(search.select(LANDSAT, [known, unknown], MAMMOTH_BBOX)) == ["LC08_known", "LC08_unknown"]


def test_bbox_overlap():
    assert search.bbox_overlap((0, 0, 2, 2), (1, 1, 3, 3)) == 1.0
    assert search.bbox_overlap((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
