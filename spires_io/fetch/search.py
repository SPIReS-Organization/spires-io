# spires_io/fetch/search.py
"""Query STAC catalogs and pick one item per acquisition.

The dedupe is two passes, ported from spipyservice ``target_chain.py``
(``filter_latest_items`` / ``coverage_deduped_by_date``): first drop superseded
reprocessings of the same satellite pass, then per calendar date keep the item
covering most of the AOI. Pass 1 must not key on the date alone: S2A and S2B can
image one tile ten minutes apart, and a date key would silently drop one.
"""
from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict

import pystac
from pystac_client import Client

from spires_io.fetch.sources import Bbox, Source, created_key, item_datetime

log = logging.getLogger(__name__)


def prefilter(source: Source, items: list[pystac.Item]) -> list[pystac.Item]:
    kept = [item for item in items if source.keep(item)]
    log.info("%s: pre-filter dropped %d of %d items", source.sensor, len(items) - len(kept), len(items))
    return kept


def latest_per_acquisition(source: Source, items: list[pystac.Item]) -> list[pystac.Item]:
    """Pass 1: one item per (acquisition, tile); the most recently processed wins."""
    latest: dict[tuple, pystac.Item] = {}
    for item in items:
        key = (source.acquisition_key(item), source.tile(item))
        if key not in latest or created_key(item) > created_key(latest[key]):
            latest[key] = item
    log.info("%s: dropped %d superseded reprocessings", source.sensor, len(items) - len(latest))
    return list(latest.values())


def bbox_overlap(a, b) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def one_per_date(source: Source, items: list[pystac.Item], bbox: Bbox) -> list[pystac.Item]:
    """Pass 2: per acquisition date keep the item covering most of the AOI.

    Lowest catalog nodata fraction wins; items without one rank below any that
    have it and are ordered by bbox overlap with the AOI among themselves.
    """
    by_date: dict[dt.date, list[pystac.Item]] = defaultdict(list)
    for item in items:
        by_date[item_datetime(item).date()].append(item)

    def rank(item):
        nodata = source.nodata_pct(item)
        return (nodata is None, nodata or 0.0, -bbox_overlap(item.bbox, bbox))

    chosen = []
    for date in sorted(by_date):
        best = min(by_date[date], key=rank)
        for item in by_date[date]:
            if item.id != best.id:
                log.info("%s: %s dropping %s in favour of %s", source.sensor, date, item.id, best.id)
        chosen.append(best)
    return chosen


def select(source: Source, items: list[pystac.Item], bbox: Bbox,
           max_cloud: float = 100.0, max_scenes: int | None = None) -> list[pystac.Item]:
    """Pre-filter, dedupe, cloud-filter, sort ascending by time, cap."""
    items = one_per_date(source, latest_per_acquisition(source, prefilter(source, items)), bbox)
    if max_cloud < 100:
        kept = [item for item in items
                if item.properties.get("eo:cloud_cover") is not None
                and float(item.properties["eo:cloud_cover"]) <= max_cloud]
        log.info("%s: cloud filter dropped %d of %d items (max %.0f%%)",
                 source.sensor, len(items) - len(kept), len(items), max_cloud)
        items = kept
    items.sort(key=item_datetime)
    return items if max_scenes is None else items[:max_scenes]
