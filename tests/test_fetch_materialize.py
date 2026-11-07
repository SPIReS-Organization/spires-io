# tests/test_fetch_materialize.py
import json
import netrc

import pytest

pytest.importorskip("pystac")
from fetch_helpers import FakeAssetServer, make_stac_item

from spires_io.fetch import materialize as mod
from spires_io.fetch.materialize import FetchError, check_earthdata_auth, materialize


@pytest.fixture
def server():
    srv = FakeAssetServer()
    yield srv
    srv.close()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(mod, "BACKOFF_S", 0.0)


def item_for(server, sizes=None):
    files = {"/scene/X_SR_B2.TIF": b"band-two", "/scene/X_MTL.txt": b"GROUP = L1\n"}
    server.files.update(files)
    assets = {"blue": server.url + "/scene/X_SR_B2.TIF", "mtl.txt": server.url + "/scene/X_MTL.txt"}
    if sizes:
        assets = {k: (v, sizes.get(k)) for k, v in assets.items()}
    return make_stac_item("X", "2024-02-14T18:23:01Z", {"eo:cloud_cover": 1.0}, assets)


def test_materialize_writes_layout_and_item_json(server, tmp_path):
    item = item_for(server)
    scene = materialize(item, tmp_path / "landsat" / "X", ["blue", "mtl.txt"])
    assert scene == tmp_path / "landsat" / "X"
    assert (scene / "X_SR_B2.TIF").read_bytes() == b"band-two"
    assert (scene / "X_MTL.txt").read_bytes() == b"GROUP = L1\n"
    assert json.loads((scene / "item.json").read_text())["id"] == "X"
    assert not (tmp_path / "landsat" / "X.partial").exists()


def test_materialize_skips_existing_without_network(server, tmp_path):
    item = item_for(server)
    scene = tmp_path / "landsat" / "X"
    scene.mkdir(parents=True)
    assert materialize(item, scene, ["blue"]) == scene
    assert sum(server.hits.values()) == 0


def test_materialize_retries_5xx_then_succeeds(server, tmp_path):
    item = item_for(server)
    server.fail_first["/scene/X_SR_B2.TIF"] = 2
    scene = materialize(item, tmp_path / "landsat" / "X", ["blue"])
    assert (scene / "X_SR_B2.TIF").read_bytes() == b"band-two"
    assert server.hits["/scene/X_SR_B2.TIF"] == 3


def test_download_retries_dropped_connection(server, tmp_path):
    item = item_for(server)
    payload = bytes(range(100))
    server.files["/scene/X_SR_B2.TIF"] = payload          # Content-Length will be 100
    server.truncate_first["/scene/X_SR_B2.TIF"] = 40      # first GET sends 40 bytes then closes
    scene = materialize(item, tmp_path / "landsat" / "X", ["blue"])
    assert (scene / "X_SR_B2.TIF").read_bytes() == payload
    assert server.hits["/scene/X_SR_B2.TIF"] == 2


def test_materialize_gives_up_after_retries_and_leaves_partial(server, tmp_path):
    item = item_for(server)
    server.fail_first["/scene/X_SR_B2.TIF"] = 99
    with pytest.raises(FetchError, match="503"):
        materialize(item, tmp_path / "landsat" / "X", ["blue"])
    assert (tmp_path / "landsat" / "X.partial").is_dir()
    assert not (tmp_path / "landsat" / "X").exists()
    assert server.hits["/scene/X_SR_B2.TIF"] == mod.ATTEMPTS


def test_materialize_404_fails_without_retry(server, tmp_path):
    item = item_for(server)
    del server.files["/scene/X_MTL.txt"]
    with pytest.raises(FetchError, match="404"):
        materialize(item, tmp_path / "landsat" / "X", ["blue", "mtl.txt"])
    assert server.hits["/scene/X_MTL.txt"] == 1


def test_materialize_size_mismatch_is_an_error(server, tmp_path):
    item = item_for(server, sizes={"blue": len(b"band-two") + 1})
    with pytest.raises(FetchError, match="catalog says 9"):
        materialize(item, tmp_path / "landsat" / "X", ["blue"])
    assert server.hits["/scene/X_SR_B2.TIF"] == 1


def test_materialize_rejects_missing_asset_key(server, tmp_path):
    item = item_for(server)
    with pytest.raises(FetchError, match="qa_pixel"):
        materialize(item, tmp_path / "landsat" / "X", ["blue", "qa_pixel"])
    assert sum(server.hits.values()) == 0


def test_materialize_replaces_stale_partial(server, tmp_path):
    item = item_for(server)
    stale = tmp_path / "landsat" / "X.partial"
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x")
    scene = materialize(item, tmp_path / "landsat" / "X", ["blue"])
    assert not (scene / "junk").exists() and not stale.exists()


class _Netrc:
    def __init__(self, hosts):
        self.hosts = hosts

    def authenticators(self, host):
        return ("user", None, "pw") if host in self.hosts else None


def test_check_earthdata_auth_passes_with_entry(monkeypatch):
    monkeypatch.setattr(netrc, "netrc", lambda: _Netrc({"urs.earthdata.nasa.gov"}))
    check_earthdata_auth()


def test_check_earthdata_auth_fails_without_entry(monkeypatch):
    monkeypatch.setattr(netrc, "netrc", lambda: _Netrc(set()))
    with pytest.raises(FetchError, match="urs.earthdata.nasa.gov"):
        check_earthdata_auth()


def test_check_earthdata_auth_fails_without_file(monkeypatch):
    def boom():
        raise FileNotFoundError
    monkeypatch.setattr(netrc, "netrc", boom)
    with pytest.raises(FetchError, match="netrc"):
        check_earthdata_auth()
