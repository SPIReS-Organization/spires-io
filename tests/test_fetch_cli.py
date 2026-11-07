import dataclasses
import datetime as dt

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


def test_non_fetch_errors_are_reported_not_raised(monkeypatch, tmp_path, capsys):
    items = {"landsat": [landsat_item(item_id="LC08_A"), landsat_item(item_id="LC09_B", platform="landsat-9",
                                                                        when="2024-02-22T18:23:01Z")]}
    monkeypatch.setattr(cli.search, "query", lambda source, *a: items[source.sensor])
    monkeypatch.setattr(cli, "check_earthdata_auth", lambda: pytest.fail("landsat needs no Earthdata auth"))
    monkeypatch.setitem(cli.SOURCES, "landsat", dataclasses.replace(
        cli.SOURCES["landsat"], make_signer=lambda: (lambda href: href)))
    done = []

    def fake_materialize(item, scene_dir, asset_keys, sign, session=None):
        if item.id == "LC08_A":
            raise OSError("disk full")
        done.append(scene_dir)
        return scene_dir

    monkeypatch.setattr(cli, "materialize", fake_materialize)
    code = cli.main(["--dest", str(tmp_path), "--aoi", "mammoth", "--sensor", "landsat"])
    assert code == 1
    assert done == [tmp_path / "landsat" / "LC09_B"]
    assert "1 scene(s) materialized, 1 failed: ['LC08_A']" in capsys.readouterr().out


def test_search_flags_reach_select(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_query(source, bbox, start, end):
        calls.append((bbox, start, end))
        if source.sensor != "landsat":
            return []
        return [landsat_item(item_id="LC08_EARLY", when="2024-02-05T18:23:01Z", cloud=90.0),
                landsat_item(item_id="LC08_MID", when="2024-02-14T18:23:01Z", cloud=5.0),
                landsat_item(item_id="LC08_LATE", when="2024-02-22T18:23:01Z", cloud=20.0)]

    monkeypatch.setattr(cli.search, "query", fake_query)
    code = cli.main(["--dest", str(tmp_path), "--bbox", "-119.1", "37.6", "-119.0", "37.7",
                     "--start", "2024-02-01", "--end", "2024-03-01", "--max-cloud", "50", "--max-scenes", "1",
                     "--dry-run"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    assert set(calls) == {((-119.1, 37.6, -119.0, 37.7), dt.date(2024, 2, 1), dt.date(2024, 3, 1))}
    assert len(lines) == 1 and "LC08_MID" in lines[0]


def test_unknown_scene_id_prefix_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="cannot infer sensor"):
        cli.main(["--dest", str(tmp_path), "--dry-run", "--scene-id", "LE07_X"])


def test_empty_cache_env_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("SPIRES_SCENE_CACHE", "")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--aoi", "mammoth", "--dry-run"])
    assert exc.value.code == 2


def test_catalog_failure_is_reported_in_summary(monkeypatch, tmp_path, capsys):
    import requests

    def fake_query(source, *a):
        if source.sensor == "landsat":
            raise requests.ConnectionError("down")
        return [hls_item("L30")]

    monkeypatch.setattr(cli.search, "query", fake_query)
    monkeypatch.setattr(cli, "check_earthdata_auth", lambda: None)
    done = []

    def fake_materialize(item, scene_dir, *a, **k):
        done.append(scene_dir)
        return scene_dir

    monkeypatch.setattr(cli, "materialize", fake_materialize)
    code = cli.main(["--dest", str(tmp_path), "--sensor", "landsat,hls_l30", "--aoi", "mammoth"])
    out = capsys.readouterr().out
    assert code == 1
    assert done == [tmp_path / "hls_l30" / "HLS.L30.T11SLB.2024045T182301.v2.0"]
    assert "landsat: query failed" in out


def test_existing_scenes_need_no_signer(monkeypatch, tmp_path, capsys):
    (tmp_path / "landsat" / "LC08_A").mkdir(parents=True)
    monkeypatch.setattr(cli.search, "query", lambda source, *a: [landsat_item(item_id="LC08_A")])
    monkeypatch.setitem(cli.SOURCES, "landsat", dataclasses.replace(
        cli.SOURCES["landsat"], make_signer=lambda: pytest.fail("signer must not be called")))
    code = cli.main(["--dest", str(tmp_path), "--aoi", "mammoth", "--sensor", "landsat"])
    assert code == 0
    assert "1 scene(s) materialized, 0 failed" in capsys.readouterr().out
