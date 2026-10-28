"""Direct COG reads for windows: one SAS token for each blob container, kept on disk until 5
minutes before it expires and renewed when a read fails with 403, and one internal block at a
time. Only Azure Blob Storage hrefs are signed."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError, WindowError
from rasterio.windows import Window, from_bounds

from mpc.http import API, agent, fetch, root, write_atomic

SAS = f"{API}/sas/v1/token"
BLOB = ".blob.core.windows.net"
MARGIN_S = 300
ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_USERAGENT": agent(),
}
_lock = threading.Lock()
_tokens: dict = {}  # the file's tokens, read once for each home


def container(href: str) -> str | None:
    """ "account/container" of an Azure Blob Storage href; None for any other href."""
    u = urlparse(href)
    if u.scheme != "https" or not (u.hostname or "").endswith(BLOB):
        return None
    return f"{u.hostname.removesuffix(BLOB)}/{u.path.split('/')[1]}"


def token(key: str, fresh: bool = False) -> str:
    path = root() / "tokens.json"
    with _lock:
        if str(path) not in _tokens:
            _tokens[str(path)] = json.loads(path.read_text()) if path.exists() else {}
        tokens = _tokens[str(path)]
        t = tokens.get(key)
        left = (datetime.fromisoformat(t["expiry"]) - datetime.now(UTC)).total_seconds() if t else 0
        if fresh or left < MARGIN_S:
            d = fetch("GET", f"{SAS}/{key}").json()
            t = tokens[key] = {"token": d["token"], "expiry": d["msft:expiry"]}
            write_atomic(path, json.dumps(tokens).encode())
        return t["token"]


def opened(item, asset: str, fn: Callable):
    """`fn(dataset)` on the asset's COG, signed; once more with a new token after a 403."""
    href = item.href(asset)
    key = container(href)
    for fresh in (False, True):
        sas = token(key, fresh) if key else ""
        url = f"{href}{'&' if '?' in href else '?'}{sas}" if key else href
        try:
            with rasterio.Env(**ENV), rasterio.open(url) as src:
                return fn(src)
        except RasterioIOError as e:
            if fresh or not key or "403" not in str(e):
                why = str(e).replace(sas, "<sas>") if sas else str(e)
                raise OSError(f"cannot read {asset} of {item.id}: {why}") from None
    raise AssertionError("unreachable")


def header(item, asset: str) -> dict:
    """The asset's grid and block layout, from its COG header."""

    def read(src):
        if src.crs is None:
            raise ValueError(f"{asset} of {item.id} has no CRS")
        bh, bw = src.block_shapes[0]
        if bh != bw or bh < 64:  # striped or odd layouts: cache 512-pixel squares instead
            bh = bw = 512
        return {
            "crs": src.crs.to_string(),
            "transform": list(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "block": bh,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "overviews": src.overviews(1),
        }

    return opened(item, asset, read)


def fetch_block(item, asset: str, g: dict, row: int, col: int) -> np.ndarray:
    """Block (`row`, `col`) of the asset: (count, h, w), smaller at the right and bottom edges."""
    b = g["block"]
    w = Window(col * b, row * b, min(b, g["width"] - col * b), min(b, g["height"] - row * b))
    return opened(item, asset, lambda src: src.read(window=w))


def overview(item, asset: str, bounds: tuple, px: int) -> np.ndarray:
    """(count, px, px) of the asset over `bounds` (W, S, E, N in its CRS), NaN off the raster
    and at nodata. A decimated read: GDAL reads the overview that fits `px`, not native blocks."""

    def read(src):
        out = np.full((src.count, px, px), np.nan, dtype=np.float32)
        full = from_bounds(*bounds, transform=src.transform)
        try:
            part = full.intersection(Window(0, 0, src.width, src.height))
        except WindowError:
            return out
        k = px / full.width
        r0 = round((part.row_off - full.row_off) * k)
        c0 = round((part.col_off - full.col_off) * k)
        h = min(max(round(part.height * k), 1), px - r0)
        w = min(max(round(part.width * k), 1), px - c0)
        a = src.read(window=part, out_shape=(src.count, h, w), masked=True,
                     resampling=Resampling.nearest)  # fmt: skip
        out[:, r0 : r0 + h, c0 : c0 + w] = a.astype(np.float32).filled(np.nan)
        return out

    return opened(item, asset, read)
