import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform

LON, LAT = 34.39767, 0.48864
CRS = "EPSG:32636"
X, Y = (v[0] for v in transform("EPSG:4326", CRS, [LON], [LAT]))
DATA = Path(__file__).with_name("data")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MPC_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def raster(path, res, size, origin, fill, block=128, count=1, dtype="uint16", overviews=()):
    """A tiled GeoTIFF, a stand-in for a COG; `fill(rows, cols)` gives the pixel values and
    `overviews` the factors of its average overviews."""
    rows, cols = np.mgrid[0:size, 0:size]
    data = np.stack([fill(rows, cols) + 1000 * b for b in range(count)]).astype(dtype)
    profile = dict(driver="GTiff", width=size, height=size, count=count, dtype=dtype, crs=CRS,
                   transform=from_origin(*origin, res, res), tiled=True, blockxsize=block,
                   blockysize=block, nodata=0)  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if overviews:
            dst.build_overviews(list(overviews), Resampling.average)
    return path


def origin_for(res, row, col):
    """A grid origin that puts the test point in pixel (row, col), near its center."""
    return (X - (col + 0.5) * res, Y + (row + 0.5) * res)


def item(iid, dt, assets, collection="fake", **props):
    poly = [[LON - 1, LAT - 1], [LON + 1, LAT - 1], [LON + 1, LAT + 1], [LON - 1, LAT + 1]]
    return {
        "id": iid,
        "collection": collection,
        "properties": {"datetime": dt, **props},
        "geometry": {"type": "Polygon", "coordinates": [poly + [poly[0]]]},
        "assets": {k: {"href": str(v)} for k, v in assets.items()},
    }


def collection(assets, cid="fake", presets=None):
    return {
        "collections": {
            cid: {
                "title": "Fake", "about": "", "keywords": [], "readable": True,
                "assets": assets, "presets": presets or {}, "time": ["2020-01-01", None],
                "bbox": [-180, -90, 180, 90], "license": "x", "hash": "",
            }
        },
        "updated": "2026-10-26T00:00:00Z",
    }  # fmt: skip


@pytest.fixture
def collections_json():
    return json.loads((DATA / "collections.json").read_text())
