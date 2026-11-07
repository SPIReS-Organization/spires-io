import dataclasses

import pytest

pytest.importorskip("pystac")
from fetch_helpers import hls_item, landsat_item

from spires_io.fetch import cli
from spires_io.fetch.materialize import FetchError


@pytest.fixture(autouse=True)
def no_cache_env(monkeypatch):
    monkeypatch.delenv("SPIRES_SCENE_CACHE", raising=False)


def test_where_is_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--dest", str(tmp_path), "--aoi", "mammoth", "--bbox", "0", "0", "1", "1"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--dest", str(tmp_path)])


def test_dest_required_unless_env(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--aoi", "mammoth", "--dry-run"])
    assert exc.value.code == 2
    monkeypatch.setenv("SPIRES_SCENE_CACHE", str(tmp_path))
    assert cli.build_parser().parse_args(["--aoi", "mammoth"]).dest == tmp_path


def test_unknown_sensor_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown sensor"):
        cli.main(["--dest", str(tmp_path), "--aoi", "mammoth", "--sensor", "modis", "--dry-run"])


def test_dry_run_prints_selection_and_writes_nothing(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_query(source, bbox, start, end):
        calls.append((source.sensor, bbox))
        return [landsat_item(item_id="LC08_A", cloud=3.0)]

    monkeypatch.setattr(cli.search, "query", fake_query)
    code = cli.main(["--dest", str(tmp_path), "--aoi", "mammoth", "--sensor", "landsat", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "landsat\tLC08_A\t2024-02-14T18:23\tcloud=3.0\ttile=042034" in out
    assert calls == [("landsat", cli.AOIS["mammoth"])]
    assert list(tmp_path.iterdir()) == []


def test_scene_id_mode_infers_sensor_and_skips_search(monkeypatch, tmp_path):
    seen = []

    def fake_fetch_by_ids(source, ids):
        seen.append((source.sensor, list(ids)))
        return []

    monkeypatch.setattr(cli.search, "fetch_by_ids", fake_fetch_by_ids)
    monkeypatch.setattr(cli.search, "query", lambda *a: pytest.fail("search must not run"))
    code = cli.main(["--dest", str(tmp_path), "--dry-run", "--scene-id",
                     "LC08_L2SP_042034_20240214_02_T1", "HLS.L30.T11SLB.2024045T182301.v2.0",
                     "LC09_L2SP_042034_20240222_02_T1"])
    assert code == 0
    assert sorted(seen) == [("hls_l30", ["HLS.L30.T11SLB.2024045T182301.v2.0"]),
                            ("landsat", ["LC08_L2SP_042034_20240214_02_T1", "LC09_L2SP_042034_20240222_02_T1"])]


def test_materialize_is_called_per_item_and_failures_set_exit_code(monkeypatch, tmp_path, capsys):
    items = {"landsat": [landsat_item(item_id="LC08_A"), landsat_item(item_id="LC09_B", platform="landsat-9",
                                                                        when="2024-02-22T18:23:01Z")]}
    monkeypatch.setattr(cli.search, "query", lambda source, *a: items[source.sensor])
    monkeypatch.setattr(cli, "check_earthdata_auth", lambda: pytest.fail("landsat needs no Earthdata auth"))
    # Source is a frozen dataclass: swap the Planetary Computer signer (network) by rebuilding the entry.
    monkeypatch.setitem(cli.SOURCES, "landsat", dataclasses.replace(
        cli.SOURCES["landsat"], make_signer=lambda: (lambda href: href)))
    done = []

    def fake_materialize(item, scene_dir, asset_keys, sign, session=None):
        if item.id == "LC09_B":
            raise FetchError("boom")
        done.append(scene_dir)
        return scene_dir

    monkeypatch.setattr(cli, "materialize", fake_materialize)
    code = cli.main(["--dest", str(tmp_path), "--aoi", "mammoth", "--sensor", "landsat"])
    assert code == 1
    assert done == [tmp_path / "landsat" / "LC08_A"]
    assert "1 scene(s) materialized, 1 failed: ['LC09_B']" in capsys.readouterr().out


def test_earthdata_check_runs_for_hls(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.search, "query", lambda source, *a: [hls_item("L30")])
    monkeypatch.setattr(cli, "check_earthdata_auth", lambda: (_ for _ in ()).throw(FetchError("no netrc")))
    with pytest.raises(SystemExit, match="no netrc"):
        cli.main(["--dest", str(tmp_path), "--aoi", "mammoth", "--sensor", "hls_l30"])
