# spires_io/fetch/cli.py
"""``spires-fetch``: search public STAC catalogs over an AOI and materialize whole scenes for the file-based adapters."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import pystac
import requests

from spires_io.fetch import search
from spires_io.fetch.materialize import FetchError, check_earthdata_auth, materialize
from spires_io.fetch.sources import AOIS, SOURCES, Source, item_datetime, sensor_from_scene_id

log = logging.getLogger("spires_io.fetch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spires-fetch",
        description="Search STAC catalogs over an AOI, dedupe reprocessings, and download whole scenes into "
                    "<dest>/<sensor>/<scene_id>/ for spires_io's raster-scene adapters.",
    )
    parser.add_argument("--dest", type=Path, default=os.environ.get("SPIRES_SCENE_CACHE") or None,
                        help="scene root — local materialization of STAC items for the file-based adapters "
                             "(default: $SPIRES_SCENE_CACHE)")
    parser.add_argument("--sensor", default=",".join(SOURCES),
                        help=f"comma-separated subset of {','.join(SOURCES)} (default: all; ignored with --scene-id)")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--aoi", choices=sorted(AOIS), help="named site")
    where.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="degrees")
    where.add_argument("--scene-id", nargs="+", metavar="ID", help="fetch exactly these ids; no search or dedupe")
    parser.add_argument("--start", type=dt.date.fromisoformat, help="YYYY-MM-DD (default: --end minus 45 days)")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--max-scenes", type=int, help="per sensor, after dedupe, earliest first")
    parser.add_argument("--max-cloud", type=float, default=100.0, help="eo:cloud_cover ceiling in percent")
    parser.add_argument("--dry-run", action="store_true", help="print the selection and exit")
    return parser


def requested_sensors(args: argparse.Namespace) -> list[str]:
    if args.scene_id:
        try:
            return sorted({sensor_from_scene_id(scene_id) for scene_id in args.scene_id})
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    sensors = [name.strip() for name in args.sensor.split(",") if name.strip()]
    unknown = [name for name in sensors if name not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown sensor(s) {unknown}; choose from {list(SOURCES)}")
    return sensors


def plan(args: argparse.Namespace) -> list[tuple[Source, list[pystac.Item]]]:
    """Resolve what to fetch: (source, items) per sensor, by search + dedupe or by explicit id."""
    if args.scene_id:
        by_sensor: dict[str, list[str]] = {}
        for scene_id in args.scene_id:
            by_sensor.setdefault(sensor_from_scene_id(scene_id), []).append(scene_id)
        return [(SOURCES[name], search.fetch_by_ids(SOURCES[name], ids)) for name, ids in by_sensor.items()]

    bbox = AOIS[args.aoi] if args.aoi else tuple(args.bbox)
    end = args.end or dt.date.today()
    start = args.start or end - dt.timedelta(days=45)
    jobs = []
    for name in requested_sensors(args):
        source = SOURCES[name]
        items = search.query(source, bbox, start, end)
        jobs.append((source, search.select(source, items, bbox, args.max_cloud, args.max_scenes)))
    return jobs


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dest is None:
        parser.error("--dest is required (or set SPIRES_SCENE_CACHE)")

    sensors = requested_sensors(args)
    if not args.dry_run and any(SOURCES[name].needs_earthdata for name in sensors):
        try:
            check_earthdata_auth()
        except FetchError as exc:
            raise SystemExit(str(exc)) from exc

    try:
        jobs = plan(args)
    except LookupError as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        for source, items in jobs:
            for item in items:
                print(f"{source.sensor}\t{item.id}\t{item_datetime(item):%Y-%m-%dT%H:%M}"
                      f"\tcloud={item.properties.get('eo:cloud_cover', '?')}\ttile={source.tile(item)}")
        return 0

    done, failed = [], []
    for source, items in jobs:
        if not items:
            log.warning("%s: nothing selected", source.sensor)
            continue
        try:
            sign = source.make_signer()
        except requests.RequestException as exc:
            log.error("%s: could not obtain signer: %s", source.sensor, exc)
            failed.extend(item.id for item in items)
            continue
        for item in items:
            scene_dir = args.dest / source.sensor / item.id
            try:
                done.append(materialize(item, scene_dir, source.assets, sign))
            except (FetchError, OSError, requests.RequestException) as exc:
                log.error("%s: %s", item.id, exc)
                failed.append(item.id)
    print(f"{len(done)} scene(s) materialized, {len(failed)} failed" + (f": {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
