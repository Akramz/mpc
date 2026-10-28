import contextlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from conftest import CRS, LAT, LON, X, Y, collection, item, origin_for, raster
from rasterio.errors import RasterioIOError
from rasterio.warp import transform
from rasterio.windows import Window

from mpc import catalog, cli, cog, nav, stac, store
from mpc import expr as ex
from mpc import reader as rd
from mpc.reader import Read, run
from mpc.units import physical


class Resp:
    def __init__(self, d):
        self.d = d

    def json(self):
        return self.d


def fake_search(monkeypatch, docs):
    def search(coll, lon, lat, dt):
        a, b = (stac.when(x) for x in dt.split("/"))
        return [i for i in map(stac.Item, docs) if i.end >= a and i.start <= b]

    monkeypatch.setattr(stac, "search", search)


def fake_point(docs):
    """The data API's point endpoint, answered from the local rasters."""
    calls = []

    def fetch(method, url, params=None, body=None, tries=4):
        calls.append(params)
        doc = docs[dict(params)["item"]]
        lon, lat = map(float, url.rsplit("/", 1)[1].split(","))
        names, values = [], []
        for a in [v for k, v in params if k == "assets"]:
            with rasterio.open(doc["assets"][a]["href"]) as src:
                xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                v = next(src.sample([(xs[0], ys[0])]))
            names += [f"{a}_b{b}" for b in range(1, len(v) + 1)]
            values += [float(x) for x in v]
        return Resp({"band_names": names, "values": values})

    fetch.calls = calls
    return fetch


def coll_of(assets, cid="fake"):
    return {"id": cid, **collection(assets, cid)["collections"][cid]}


def test_catalog_records_use_exact_asset_names(collections_json):
    cols = {c["id"]: c for c in collections_json}
    s2 = catalog.compact(cols["sentinel-2-l2a"], {})
    assert {"B04", "B8A", "SCL", "visual_b1"} <= set(s2["assets"])
    assert s2["assets"]["B11"]["res"] == 20
    ndvi = catalog.preset_exprs(
        "expression=(image_b4 - image_b1)/(image_b4 + image_b1)&rescale=-1,1"
    )
    naip = catalog.compact(cols["naip"], {"NDVI": ndvi})
    assert list(naip["assets"]) == ["image_b1", "image_b2", "image_b3", "image_b4"]
    assert naip["presets"]["NDVI"] == ["(image_b4-image_b1)/(image_b4+image_b1)"]
    cir = catalog.preset_exprs("assets=image&asset_bidx=image|4,1,2&color_formula=x")
    assert cir == ["image_b4", "image_b1", "image_b2"]
    lulc = catalog.compact(cols["io-lulc-annual-v02"], {})
    assert lulc["assets"]["data"]["classes"]["5"] == "Crops"
    red = catalog.compact(cols["landsat-c2-l2"], {})["assets"]["red"]
    assert (red["scale"], red["offset"], red["nodata"]) == (2.75e-05, -0.2, 0)


def test_search_shares_a_cell_and_keeps_footprints_that_hold_the_point(monkeypatch):
    bodies = []
    far = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    feats = [
        item("in", "2023-05-16T07:46:19Z", {}),
        {**item("out", "2023-05-16T07:46:19Z", {}), "geometry": far},
    ]

    def cached_json(method, url, body=None, ttl=None, params=None):
        bodies.append(body)
        return {"features": feats, "links": []}

    monkeypatch.setattr(stac, "cached_json", cached_json)
    dt = stac.interval(date(2023, 5, 16), date(2023, 5, 16))
    assert [i.id for i in stac.search("fake", LON, LAT, dt)] == ["in"]
    stac.search("fake", LON + 0.001, LAT + 0.001, dt)
    assert bodies[0]["bbox"] == bodies[1]["bbox"]


def test_footprints_across_the_antimeridian_hold_only_their_own_points():
    ring = [[179, 0], [-174, 0], [-174, 8], [179, 8], [179, 0]]  # UTM zone 01N, wrapped
    wrapped = {"type": "Polygon", "coordinates": [ring]}
    assert stac.contains(wrapped, 179.5, 4) and stac.contains(wrapped, -175, 4)
    assert not stac.contains(wrapped, 34.4, 0.5)


def test_an_annual_map_ending_on_january_1_is_labelled_by_its_own_year():
    i = stac.Item(item("Y", None, {}, start_datetime="2023-01-01T00:00:00Z",
                       end_datetime="2024-01-01T00:00:00Z"))  # fmt: skip
    assert i.label == "2023-01-01/2023-12-31"


def test_period_keeps_one_item_per_acquisition(monkeypatch):
    docs = [
        item("T1_a", "2023-05-01T07:46:19Z", {}),
        item("T2_a", "2023-05-01T07:46:19Z", {}),  # the same take on the next tile
        item("T1_b", "2023-05-31T07:46:21Z", {}),
        item("T1_c", "2023-06-01T07:46:19Z", {}),
        item(
            "Y2022",
            None,
            {},
            start_datetime="2022-05-01T00:00:00Z",
            end_datetime="2023-05-01T00:00:00Z",
        ),
    ]
    fake_search(monkeypatch, docs)
    got = stac.items("fake", LON, LAT, period=("2023-05-01", "2023-05-31"))
    assert [i.id for i in got] == ["T1_a", "T1_b"]  # as mpc search lists them


def test_a_date_takes_only_the_image_on_that_day(monkeypatch):
    days = [("a", "2023-05-16"), ("b", "2023-05-16"), ("c", "2023-05-21"), ("d", "2023-09-01")]
    fake_search(monkeypatch, [item(i, f"{d}T10:0{k}:00Z", {}) for k, (i, d) in enumerate(days)])
    assert [i.id for i in stac.items("fake", LON, LAT, dates=("2023-05-16",))] == ["a"]
    with pytest.raises(LookupError, match="on 2023-05-15; mpc search"):
        stac.items("fake", LON, LAT, dates=("2023-05-15", "2023-05-21"))
    with pytest.raises(ValueError, match="give dates or a period"):
        stac.items("fake", LON, LAT)  # no date is an error, never the latest

    year = item("Y", None, {}, start_datetime="2023-01-01T00:00:00Z",
                end_datetime="2024-01-01T00:00:00Z")  # fmt: skip
    fake_search(monkeypatch, [year])
    assert [i.id for i in stac.items("fake", LON, LAT, dates=("2023-12-31",))] == ["Y"]
    with pytest.raises(LookupError):
        stac.items("fake", LON, LAT, dates=("2024-01-01",))


def test_search_prints_shared_properties_once_and_the_rest_as_columns(monkeypatch, capsys):
    docs = [
        item("S1", "2023-05-06T07:46:19Z", {}, **{"eo:cloud_cover": 2.902316, "proj:epsg": 32636}),
        item("S1b", "2023-05-06T07:46:19Z", {}, **{"eo:cloud_cover": 2.9, "proj:epsg": 32636}),
        item("S2", "2023-05-11T07:46:19Z", {}, **{"eo:cloud_cover": 15.07, "proj:epsg": 32636,
                                                  "instruments": ["msi"]}),
    ]  # fmt: skip
    fake_search(monkeypatch, docs)
    monkeypatch.setattr(catalog, "get", lambda c: {"id": c})
    args = ["search", "fake", "--location", f"{LON},{LAT}", "--period", "2023-05-01", "2023-05-31"]
    cli.main(args)
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["proj:epsg", "32636"]
    assert out[2].split() == ["date", "eo:cloud_cover", "item"]
    assert out[3].split() == ["2023-05-06", "2.902", "S1"] and len(out) == 5
    cli.main([*args, "--json"])
    rec = json.loads(capsys.readouterr().out)[1]
    assert rec["item"] == "S2" and rec["instruments"] == ["msi"] and rec["date"] == "2023-05-11"


def test_window_fetches_each_block_once_and_sub_windows_hit_the_store(tmp_path, monkeypatch, home):
    path = raster(
        tmp_path / "b04.tif", 10, 400, origin_for(10, 200, 200), lambda r, c: r * 100 + c + 1
    )
    it = stac.Item(item("A", "2023-05-16T07:46:19Z", {"B04": path}))
    fetched, orig = [], cog.fetch_block
    monkeypatch.setattr(cog, "fetch_block", lambda *a: fetched.append(a[3:]) or orig(*a))
    big = store.window(it, "B04", 100, 100, 101, 101)  # four 128-pixel blocks
    with rasterio.open(path) as src:
        want = src.read(1, window=Window(100, 100, 101, 101))
    assert np.array_equal(big[0], want) and len(fetched) == 4
    small = store.window(it, "B04", 130, 120, 11, 11)
    assert len(fetched) == 4 and np.array_equal(small[0], want[30:41, 20:31])
    edge = store.window(it, "B04", 395, 395, 10, 10)
    assert np.isnan(edge[0, 5:]).all() and np.isfinite(edge[0, :5, :5]).all()
    assert store.ls()[0][1] == "A" and store.usage()[0][0] == "fake"
    assert store.prune(collection="other")[0] == 0
    assert store.prune(collection="fake")[0] == len(fetched)
    assert not list((home / "store").rglob("*.tif"))


def test_coarse_bands_repeat_on_the_fine_grid(tmp_path):
    ox, oy = origin_for(10, 100, 100)
    fine = raster(tmp_path / "f.tif", 10, 200, (ox, oy), lambda r, c: r * 100 + c + 1)
    coarse = raster(tmp_path / "c.tif", 20, 101, (ox - 10, oy + 10), lambda r, c: r * 100 + c + 1)
    coll = coll_of({"F": {"about": "", "res": 10}, "C": {"about": "", "res": 20}})
    it = stac.Item(item("A", "2023-05-16T07:46:19Z", {"F": fine, "C": coarse}))
    notes = []
    [(arrays, crs, t)] = rd.native(coll, [it], ["C", "F"], LON, LAT, 50, 1025, notes)
    rows = np.arange(95, 106)
    big = (rows + 1) // 2  # the coarse grid starts one fine pixel up and left
    assert np.array_equal(arrays["C"], big[:, None] * 100 + big[None, :] + 1)
    assert arrays["F"][5, 5] == 100 * 100 + 100 + 1
    assert crs == CRS and (t.c, t.f) == pytest.approx((ox + 950, oy - 950))
    assert notes == ["C is 20 m; each of its pixels repeats 2 x 2 on the 10 m grid"]


def test_points_come_from_the_api_once_then_from_the_store(tmp_path, monkeypatch):
    path = raster(
        tmp_path / "b04.tif", 10, 200, origin_for(10, 100, 100), lambda r, c: r * 100 + c + 1
    )
    doc = item("A", "2023-05-16T07:46:19Z", {"B04": path})
    it = stac.Item(doc)
    api = fake_point({"A": doc})
    monkeypatch.setattr(rd, "fetch", api)
    assert rd.point_values(it, ["B04"], LON, LAT) == {"B04": 10101.0}
    assert rd.point_values(it, ["B04"], LON, LAT) == {"B04": 10101.0}
    store.window(it, "B04", 0, 0, 200, 200)
    lon, lat = (v[0] for v in transform(CRS, "EPSG:4326", [X + 30], [Y]))
    assert rd.point_values(it, ["B04"], lon, lat) == {"B04": 10104.0}
    assert len(api.calls) == 1


def test_sentinel2_reflectance_follows_the_processing_baseline():
    coll = {"id": "sentinel-2-l2a", "assets": {"B04": {}, "SCL": {}}}

    def s2(baseline):
        return stac.Item(item("s", "2023-05-16T00:00:00Z", {}, "sentinel-2-l2a",
                              **{"s2:processing_baseline": baseline}))  # fmt: skip

    assert physical(coll, s2("05.09"), "B04", 2000) == pytest.approx(0.1)
    assert physical(coll, s2("03.01"), "B04", 2000) == pytest.approx(0.2)
    assert np.isnan(physical(coll, s2("05.09"), "B04", 0))
    assert physical(coll, s2("05.09"), "SCL", 4) == 4


def test_tokens_renew_near_expiry_and_after_a_403(monkeypatch):
    issued = []

    def fetch(method, url, **kw):
        issued.append(url)
        left = timedelta(minutes=3 if len(issued) == 1 else 45)
        return Resp(
            {"token": f"t{len(issued)}", "msft:expiry": (datetime.now(UTC) + left).isoformat()}
        )

    monkeypatch.setattr(cog, "fetch", fetch)
    assert cog.token("acct/cont") == "t1"  # expires within the margin: the next call renews it
    assert cog.token("acct/cont") == "t2" and cog.token("acct/cont") == "t2"
    urls = []

    def fake_open(url):
        urls.append(url)
        if len(urls) == 1:
            raise RasterioIOError("HTTP response code: 403")
        return contextlib.nullcontext("src")

    monkeypatch.setattr(cog.rasterio, "open", fake_open)
    hrefs = {"B04": "https://acct.blob.core.windows.net/cont/b04.tif", "B08": "https://x.test/b"}
    it = stac.Item(item("A", "2023-05-16T07:46:19Z", hrefs, "c"))
    assert cog.opened(it, "B04", lambda src: src) == "src"
    assert urls[0].endswith("?t2") and urls[1].endswith("?t3")
    assert issued[-1].endswith("/token/acct/cont")  # a token for the blob container
    cog.opened(it, "B08", lambda src: src)
    assert urls[-1] == "https://x.test/b"  # other hosts never see a token

    def denied(url):
        raise RasterioIOError(f"'/vsicurl/{url}': HTTP response code: 403")

    monkeypatch.setattr(cog.rasterio, "open", denied)
    with pytest.raises(OSError, match="cannot read B04 of A") as e:
        cog.opened(it, "B04", lambda src: src)
    assert "<sas>" in str(e.value) and "?t" not in str(e.value)  # errors never show the token


def test_read_end_to_end_table_view_and_save(tmp_path, monkeypatch):
    docs = []
    for k, dt in enumerate(["2023-05-11T08:00:00Z", "2023-05-16T08:00:00Z"]):
        o = origin_for(10, 100, 100)
        b04 = raster(tmp_path / f"b04_{k}.tif", 10, 200, o, lambda r, c, k=k: 1000 + 100 * k + c)
        b08 = raster(tmp_path / f"b08_{k}.tif", 10, 200, o, lambda r, c, k=k: 3000 + 100 * k + c)
        docs.append(item(f"S{k}", dt, {"B04": b04, "B08": b08}))
    unit = {"about": "", "res": 10, "scale": 1e-4, "nodata": 0}
    cat = collection({"B04": unit, "B08": unit})
    fake_search(monkeypatch, docs)
    monkeypatch.setattr(rd, "fetch", fake_point({d["id"]: d for d in docs}))

    ndvi = "(B08-B04)/(B08+B04)"
    res = run(Read("fake", LON, LAT, (ndvi,), period=("2023-05-11", "2023-05-16")), cat)
    assert res.values[:, 0] == pytest.approx([2000 / 4200, 2000 / 4400])
    assert rd.table(res).splitlines()[1].startswith("2023-05-11")

    out = tmp_path / "out.tif"
    req = Read("fake", LON, LAT, ("B08", "B04", "B04"), ("2023-05-16",), buffer_m=50, save=str(out))
    res = run(req, cat)
    assert res.view.exists() and [i.id for i in res.items] == ["S1"]
    with rasterio.open(out) as src:
        assert src.descriptions == ("2023-05-16_B08", "2023-05-16_B04", "2023-05-16_B04")
        assert src.res == (10, 10) and src.shape == (11, 11)
        assert src.read(1)[5, 5] == pytest.approx(0.32)
        assert json.loads(src.tags()["times"]) == ["2023-05-16"]

    with pytest.raises(ValueError, match="unknown asset 'nir'"):
        run(Read("fake", LON, LAT, ("(nir-B04)",), ("2023-05-16",)), cat)
    with pytest.raises(ValueError, match="give --expr"):
        run(Read("fake", LON, LAT, (), ("2023-05-16",)), cat)


def test_negative_longitudes_parse():
    argv = "read naip --location -76.05,38.77 --expr image_b1 --dates 2023-05-16".split()
    a = cli.parser().parse_args(cli.joined(argv))
    assert a.location == (-76.05, 38.77)


def test_points_fall_back_to_the_cog_when_the_api_fails(tmp_path, monkeypatch):
    here = raster(
        tmp_path / "here.tif", 10, 200, origin_for(10, 100, 100), lambda r, c: r * 100 + c + 1
    )
    away = raster(tmp_path / "away.tif", 10, 50, (X + 1e5, Y), lambda r, c: r + 1)  # off the point

    def broken(method, url, **kw):
        req = httpx.Request(method, url)
        raise httpx.HTTPStatusError("500", request=req, response=httpx.Response(500, request=req))

    monkeypatch.setattr(rd, "fetch", broken)
    hit = stac.Item(item("here", "2023-05-16T07:46:19Z", {"data": here}))
    miss = stac.Item(item("away", "2023-05-16T07:46:19Z", {"data": away}))
    assert rd.point_values(hit, ["data"], LON, LAT) == {"data": 10101.0}
    assert np.isnan(rd.point_values(miss, ["data"], LON, LAT)["data"])


def checkerboard(path, size=1024):
    """1 and 3 on alternate native pixels: every average overview is 2, a native read never is."""
    o = origin_for(10, size // 2, size // 2)
    return raster(path, 10, size, o, lambda r, c: 1 + 2 * ((r + c) % 2), overviews=(2, 4, 8))


def test_paths_halve_the_scene_and_percent_maps_back_to_the_view(tmp_path, monkeypatch):
    doc = item("S1", "2023-05-16T08:00:00Z", {"B04": checkerboard(tmp_path / "b.tif")})
    cat = collection({"B04": {"about": "", "res": 10}})
    monkeypatch.setattr(catalog, "load", lambda: cat)
    v = nav.View(doc, (LON, LAT), "", ("B04",))
    g = nav.grid(v)
    x0, y0 = g["transform"][2], g["transform"][5]
    assert nav.box(v, g) == (x0, y0, 10240) and nav.pixels(v, g) == 128
    assert nav.level(v, g, 128) == 8
    v = nav.View(doc, (LON, LAT), nav.parse(" 2/ 3 "), ("B04",))
    assert v.path == "2/3" and nav.box(v, g) == (x0 + 5120, y0 - 2560, 2560)
    assert nav.pixels(v, g) == 128 and nav.level(v, g, 128) == 2
    for x, y in [(0, 0), (100, 100), (50, 25)]:
        lon, lat = nav.lonlat(v, x, y)
        xs, ys = transform("EPSG:4326", CRS, [lon], [lat])
        assert (xs[0], ys[0]) == pytest.approx((x0 + 5120 + 25.6 * x, y0 - 2560 - 25.6 * y))
    deep = nav.View(doc, (LON, LAT), "1/1/1/1", ("B04",))
    assert nav.pixels(deep, g) == 64 and nav.level(deep, g, 64) == 1
    with pytest.raises(ValueError, match="under 64 native pixels"):
        nav.show(nav.View(doc, (LON, LAT), "1/1/1/1/1", ("B04",)))
    with pytest.raises(ValueError, match="digits 1-4"):
        nav.parse("2/5")


def test_views_read_overviews_and_read_at_names_its_view(tmp_path, monkeypatch, capsys, home):
    path = checkerboard(tmp_path / "b04.tif")
    doc = item("S1", "2023-05-16T08:00:00Z", {"B04": path})
    cat = collection({"B04": {"about": "", "res": 10, "nodata": 0}})
    monkeypatch.setattr(catalog, "load", lambda: cat)
    fake_search(monkeypatch, [doc])
    monkeypatch.setattr(rd, "fetch", fake_point({"S1": doc}))
    it = stac.Item(doc)
    with rasterio.open(path) as src:
        w, s_, e, n = src.bounds
    assert (cog.overview(it, "B04", (w, s_, e, n), 128) == 2).all()  # the 8x overview
    native = cog.overview(it, "B04", (w, n - 1280, w + 1280, n), 128)
    assert set(np.unique(native)) == {1, 3}
    half = cog.overview(
        it, "B04", (w - 5120, s_, e - 5120, n), 64
    )  # the left half is off the raster
    assert np.isnan(half[0, :, :32]).all() and (half[0, :, 32:] == 2).all()

    where = ["--location", f"{LON},{LAT}"]
    with pytest.raises(SystemExit):
        cli.main(["navigate", "fake", *where, "--date", "2023-05-16"])  # what should it show?
    assert "give --expr" in capsys.readouterr().err
    cli.main(["navigate", "fake", *where, "--date", "2023-05-16", "--expr", "B04", "--json"])
    whole = json.loads(capsys.readouterr().out)
    assert 'path "" · the whole scene' in whole["caption"] and "[8x]" in whole["caption"]
    cli.main(["navigate", "--view", whole["view"], "--path", "2/3", "--json"])
    part = json.loads(capsys.readouterr().out)
    assert part["path"] == "2/3" and part["view"] != whole["view"]
    for v in (whole, part):  # each view keeps its own frame: no shared current view
        argv = ["read", "fake", "--view", v["view"], "--at", "50,50", "--expr", "B04"]
        cli.main([*argv, "--dates", "2023-05-16"])
        first = capsys.readouterr().out.splitlines()[0]
        lon, lat = nav.lonlat(nav.load(v["view"]), 50, 50)
        assert first.endswith(f"is lon {lon:.5f}, lat {lat:.5f}.") and v["view"] in first
    for argv in (
        ["navigate", "--path", "2/3"],  # which view?
        ["read", "fake", "--at", "50,50", "--expr", "B04", "--dates", "2023-05-16"],
        "read fake --view 0123456789ab --at 50,50 --expr B04 --dates 2023-05-16".split(),
        ["navigate", "fake", *where, "--date", "2023-05-17", "--expr", "B04"],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)
    err = capsys.readouterr().err
    assert "or --view ID" in err and "give --view ID" in err and "no view '0123456789ab'" in err
    assert "on 2023-05-17; mpc search" in err


def test_a_large_store_says_so(monkeypatch, capsys):
    monkeypatch.setattr(cli, "STORE_WARN_BYTES", -1)  # an empty store is over the line
    monkeypatch.setattr(cli, "cmd_read", lambda a: None)
    cli.main(["read", "fake", "--location", "0,0", "--expr", "B04", "--dates", "2023-05-16"])
    assert "mpc store du shows what uses it" in capsys.readouterr().err


def test_skill_frontmatter_is_strict_yaml():
    text = (Path(cli.__file__).parent.parent / "plugin" / "skills" / "mpc" / "SKILL.md").read_text()
    head = text.split("---")[1].strip().splitlines()
    for line in head:
        key, _, value = line.partition(": ")
        assert ": " not in value and not value.startswith(("`", "*", "&", "!")), key


def test_store_keeps_collections_apart():
    a, b = (stac.Item(item("37M-2020", "2020-06-01T00:00:00Z", {}, c)) for c in ("lulc", "lulc9"))
    store.save_points(a, {"data": 6.0}, LON, LAT)
    assert store.stored_point(a, "data", LON, LAT) == 6.0
    assert store.stored_point(b, "data", LON, LAT) is None


def test_expressions_take_asset_names_that_are_not_python_names():
    bands = {
        "250m_16_days_EVI": np.array([2.0]),
        "burn-severity": np.array([3.0]),
        "B08": np.ones(1),
    }
    assert ex.names("250m_16_days_EVI * 2 - burn-severity", bands) == {
        "250m_16_days_EVI",
        "burn-severity",
    }
    assert ex.evaluate("burn-severity - B08", bands)[0] == 2.0
    assert ex.names("burn-severity", {}) == {"burn", "severity"}  # unknown: plain subtraction


def test_buffer_is_meters_on_a_grid_in_degrees():
    res = 1 / 12000
    g = {"crs": "EPSG:4326", "transform": [res, 0, 34, 0, -res, 1], "width": 12000, "height": 12000}
    p = rd.plan({"data": g}, 34.4, 0.49, 100, 1025, [])
    assert p["n"] == 2 * round(100 / (res * 111_320)) + 1  # about 9.3 m pixels at the equator


def test_expected_failures_are_one_line_errors(monkeypatch, capsys):
    def down(a):
        raise httpx.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setattr(cli, "cmd_read", down)
    with pytest.raises(SystemExit) as e:
        cli.main(["read", "fake", "--location", "0,0", "--expr", "B04", "--dates", "2023-05-16"])
    assert e.value.code == 1 and capsys.readouterr().err == "error: [Errno 61] Connection refused\n"


def test_a_picture_of_too_many_dates_fails_before_any_pixel_read(monkeypatch):
    docs = [item(f"S{d}", f"2023-05-{d:02d}T08:00:00Z", {}) for d in range(1, 14)]
    fake_search(monkeypatch, docs)
    monkeypatch.setattr(rd, "native", lambda *a: pytest.fail("read pixels"))
    cat = collection({"B04": {"about": "", "res": 10}})
    with pytest.raises(ValueError, match="13 dates; an image shows at most 12"):
        run(Read("fake", LON, LAT, ("B04",), period=("2023-05-01", "2023-05-31"), buffer_m=50), cat)


def test_comparisons_give_one_zero_or_nan_where_data_is_missing():
    bands = {"SCL": np.array([6.0, 4.0, np.nan]), "B08": np.array([0.3, 0.5, 0.1])}
    assert np.array_equal(ex.evaluate("SCL==6", bands), [1, 0, np.nan], equal_nan=True)
    assert np.array_equal(ex.evaluate("(SCL!=6)*B08", bands)[:2], np.float32([0, 0.5]))
    assert np.array_equal(ex.evaluate("0.2 < B08 <= 0.5", bands), [1, 1, 0])


def test_the_plugin_the_package_and_the_skill_name_one_version():
    import tomllib

    root = Path(cli.__file__).parent.parent
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    plugin = json.loads((root / "plugin" / ".claude-plugin" / "plugin.json").read_text())
    assert plugin["version"] == version
    assert f"mpc-cli=={version}" in (root / "plugin" / "skills" / "mpc" / "SKILL.md").read_text()
