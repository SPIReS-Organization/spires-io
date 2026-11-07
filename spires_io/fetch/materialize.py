# spires_io/fetch/materialize.py
"""Download a STAC item's selected assets into the scene-directory layout the adapters read."""
from __future__ import annotations

import json
import logging
import netrc
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable

import pystac
import requests

from spires_io.fetch.sources import asset_filename

log = logging.getLogger(__name__)

EARTHDATA_HOST = "urs.earthdata.nasa.gov"
ATTEMPTS = 4          # first try + 3 retries
BACKOFF_S = 2.0
CHUNK = 1 << 20


class FetchError(RuntimeError):
    """A scene could not be materialized; its ``.partial`` directory is left in place for inspection."""


def check_earthdata_auth() -> None:
    """Fail fast if ~/.netrc cannot authenticate to Earthdata; LP DAAC otherwise 301s to a login page mid-download."""
    try:
        auth = netrc.netrc().authenticators(EARTHDATA_HOST)
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    if auth is None:
        raise FetchError(
            f"~/.netrc has no usable entry for {EARTHDATA_HOST}; HLS downloads need Earthdata Login. "
            f"Add: machine {EARTHDATA_HOST} login <user> password <password>"
        )


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)):
        return True
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code >= 500
    return False


def download(session: requests.Session, url: str, path: Path, expected_size: int | None = None) -> None:
    """Stream ``url`` to ``path``. Retries connection drops (incl. mid-body), timeouts and 5xx, up to ATTEMPTS attempts in total."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with session.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with open(path, "wb") as fh:
                    for chunk in response.iter_content(CHUNK):
                        fh.write(chunk)
            size = path.stat().st_size
            if expected_size is not None and size != expected_size:
                raise FetchError(f"{path.name}: got {size} bytes, catalog says {expected_size}")
            return
        except (requests.RequestException, FetchError) as exc:
            if attempt == ATTEMPTS or not _retryable(exc):
                raise FetchError(f"{url}: {exc}") from exc
            log.warning("%s: attempt %d/%d failed (%s); retrying", path.name, attempt, ATTEMPTS, exc)
            time.sleep(BACKOFF_S * attempt)


def materialize(item: pystac.Item, scene_dir: Path, asset_keys: Iterable[str],
                sign: Callable[[str], str] = lambda href: href,
                session: requests.Session | None = None) -> Path:
    """Write ``asset_keys`` of ``item`` plus ``item.json`` into ``scene_dir`` atomically; no-op if it already exists.

    Files land in ``<scene_dir>.partial/`` and are renamed into place only
    after ``item.json`` is written, so a complete directory is always a
    complete scene. A stale ``.partial`` from an earlier failed run is removed
    first; a new one is left behind on failure.
    """
    if scene_dir.is_dir():
        log.info("%s: already present, skipping", scene_dir.name)
        return scene_dir
    asset_keys = list(asset_keys)
    missing = [key for key in asset_keys if key not in item.assets]
    if missing:
        raise FetchError(f"{item.id}: assets {missing} not in item; available: {sorted(item.assets)}")

    session = session or requests.Session()
    partial = scene_dir.with_name(scene_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    for key in asset_keys:
        asset = item.assets[key]
        target = partial / asset_filename(asset.href)
        log.info("%s: %s -> %s", item.id, key, target.name)
        download(session, sign(asset.href), target, asset.extra_fields.get("file:size"))

    (partial / "item.json").write_text(json.dumps(item.to_dict(include_self_link=False, transform_hrefs=False), indent=2))
    os.rename(partial, scene_dir)
    return scene_dir
