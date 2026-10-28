"""Against the live Planetary Computer: `MPC_LIVE=1 uv run pytest tests/test_live.py`."""

import os
import time
from dataclasses import replace

import numpy as np
import pytest
from rasterio.warp import transform
from rasterio.windows import Window

from mpc import cog, nav
from mpc.reader import Read, run

pytestmark = pytest.mark.skipif(not os.environ.get("MPC_LIVE"), reason="set MPC_LIVE=1")
PLACES = {  # the article's READ demo
    "maize plot": (34.39767, 0.48864),
    "forest": (34.46716, 0.92971),
    "homestead": (34.51965, 0.73515),
    "wetland": (34.79602, 0.58665),
}


def reflectance(item, dn):
    offset = 1000 if item.props["s2:processing_baseline"] >= "04.00" else 0
    dn = np.asarray(dn, dtype=np.float64)
    return np.where(dn == 0, np.nan, (dn - offset) / 1e4)


@pytest.mark.parametrize("lon,lat", PLACES.values(), ids=PLACES)
def test_values_and_windows_match_plain_rasterio_reads(lon, lat):
    res = run(Read("sentinel-2-l2a", lon, lat, ("B04", "B08"), ("2023-02-15", "2023-08-14")))
    for t, item in enumerate(res.items):
        for j, asset in enumerate(("B04", "B08")):

            def sample(src):
                xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
                return next(src.sample([(xs[0], ys[0])]))[0]

            want = reflectance(item, cog.opened(item, asset, sample))
            assert res.values[t, j] == pytest.approx(want, nan_ok=True)

    alone = run(Read("sentinel-2-l2a", lon, lat, ("B11",), ("2023-05-16",), buffer_m=100))
    mixed = run(
        Read("sentinel-2-l2a", lon, lat, ("B11", "B08", "B04"), ("2023-05-16",), buffer_m=100)
    )
    item = alone.items[0]

    def window(src):
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        r, c = src.index(xs[0], ys[0])
        return src.read(1, window=Window(c - 6, r - 6, 13, 13))

    b11 = reflectance(item, cog.opened(item, "B11", window))
    # alone, B11 keeps its own 20 m grid: 5 pixels each side of the point's pixel
    assert alone.planes.shape[-2:] == (11, 11) and abs(alone.transform.a) == 20
    assert np.allclose(alone.planes[0, 0], b11[1:12, 1:12], equal_nan=True)
    # beside 10 m bands, each 20 m pixel repeats 2 x 2 on the 10 m grid
    got = mixed.planes[0, 0]
    assert got.shape == (21, 21) and abs(mixed.transform.a) == 10
    big = b11.repeat(2, 0).repeat(2, 1)
    assert any(
        np.allclose(got, big[oy : oy + 21, ox : ox + 21], equal_nan=True)
        for oy in (2, 3)  # the point's 10 m pixel lies inside its 20 m pixel
        for ox in (2, 3)
    )


def test_a_view_is_cheap_and_read_at_reads_the_spot_it_names(tmp_path, monkeypatch):
    monkeypatch.setenv("MPC_HOME", str(tmp_path))
    lon, lat = PLACES["maize plot"]
    v = nav.start("sentinel-2-l2a", lon, lat, "2023-05-16", ("B04", "B03", "B02"))
    t = time.time()
    png, text = nav.show(v)
    assert time.time() - t < 5 and png.exists()
    g = nav.grid(v)
    f = max(g["overviews"])
    assert f"[{f}x]" in text and nav.pixels(v, g) == 10980 // f
    v = replace(v, path="2/3/1")
    nav.show(v)
    lon2, lat2 = nav.lonlat(v, 62, 35)
    res = run(Read("sentinel-2-l2a", lon2, lat2, ("B04",), ("2023-05-16",)))
    item = res.items[0]

    def sample(src):
        xs, ys = transform("EPSG:4326", src.crs, [lon2], [lat2])
        return next(src.sample([(xs[0], ys[0])]))[0]

    assert res.values[0, 0] == pytest.approx(reflectance(item, cog.opened(item, "B04", sample)))
