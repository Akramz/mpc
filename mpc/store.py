"""The local store: every block a window read fetched, as one GeoTIFF on the COG's own block
grid, and every point value, indexed in SQLite by item, asset and place. Nothing expires; only
`prune` removes data."""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mpc.http import count, root, write_atomic

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (            -- item ids repeat across collections
  collection text, item text, datetime text,
  PRIMARY KEY (collection, item)
);
CREATE TABLE IF NOT EXISTS assets (           -- the grid of one asset of one item
  collection text, item text, asset text,
  grid text,                                  -- JSON: crs, transform, width, height, block, ...
  PRIMARY KEY (collection, item, asset)
);
CREATE TABLE IF NOT EXISTS blocks (           -- the file: store/<collection>/<item>/<asset>/
  collection text, item text, asset text, row int, col int,
  minx real, miny real, maxx real, maxy real, -- WGS84
  bytes int,
  PRIMARY KEY (collection, item, asset, row, col)
);
CREATE TABLE IF NOT EXISTS points (           -- data API values, for reads without a buffer
  collection text, item text, name text, lon real, lat real, value real,
  PRIMARY KEY (collection, item, name, lon, lat)
);
"""
_local = threading.local()
_init = threading.Lock()
_pool: list[ThreadPoolExecutor] = []


def db() -> sqlite3.Connection:
    path = root() / "store.sqlite"
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = _local.conns = {}
    if str(path) not in conns:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _init:  # threads that open a new store at once would lock each other out
            con = sqlite3.connect(path, timeout=30, isolation_level=None)
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)
        conns[str(path)] = con
    return conns[str(path)]


def pool() -> ThreadPoolExecutor:
    """Block fetches, shared by every window; its tasks never wait on the pool themselves."""
    if not _pool:
        _pool.append(ThreadPoolExecutor(16, thread_name_prefix="mpc-blocks"))
    return _pool[0]


def remember(item) -> None:
    db().execute(
        "INSERT OR IGNORE INTO items VALUES (?, ?, ?)",
        (item.collection, item.id, item.datetime.isoformat()),
    )


def cached_grid(item, asset: str) -> dict | None:
    row = (
        db()
        .execute(
            "SELECT grid FROM assets WHERE collection = ? AND item = ? AND asset = ?",
            (item.collection, item.id, asset),
        )
        .fetchone()
    )
    return json.loads(row[0]) if row else None


def grid(item, asset: str, need: tuple[str, ...] = ()) -> dict:
    """The asset's grid; read again when a stored one lacks a key in `need`."""
    g = cached_grid(item, asset)
    if g is None or any(k not in g for k in need):
        from mpc import cog

        g = cog.header(item, asset)
        count("requests")
        remember(item)
        db().execute(
            "INSERT OR REPLACE INTO assets VALUES (?, ?, ?, ?)",
            (item.collection, item.id, asset, json.dumps(g)),
        )
    return g


def block_path(collection: str, item: str, asset: str, row: int, col: int) -> Path:
    for part in (collection, item, asset):  # they come from STAC: keep them inside the store
        if part in ("", ".", "..") or "/" in part or "\\" in part:
            raise ValueError(f"{part!r} cannot name a folder in the store")
    return root() / "store" / collection / item / asset / f"r{row}_c{col}.tif"


def stored_block(item, asset: str, row: int, col: int) -> Path | None:
    hit = (
        db()
        .execute(
            "SELECT 1 FROM blocks WHERE collection = ? AND item = ? AND asset = ? AND row = ? "
            "AND col = ?",
            (item.collection, item.id, asset, row, col),
        )
        .fetchone()
    )
    path = block_path(item.collection, item.id, asset, row, col)
    return path if hit and path.exists() else None


def read_block(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read()


def block(item, asset: str, g: dict, row: int, col: int) -> np.ndarray:
    path = stored_block(item, asset, row, col)
    if path is not None:
        count("hits")
        return read_block(path)
    from mpc import cog

    arr = cog.fetch_block(item, asset, g, row, col)
    count("requests")
    save_block(item, asset, g, row, col, arr)
    return arr


def save_block(item, asset, g, row, col, arr) -> None:
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine, array_bounds
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, transform

    b, (n, h, w) = g["block"], arr.shape
    t = transform(Window(col * b, row * b, w, h), Affine(*g["transform"]))
    profile = dict(
        driver="GTiff", width=w, height=h, count=n, dtype=arr.dtype, crs=g["crs"],
        transform=t, nodata=g["nodata"], compress="deflate",
    )  # fmt: skip
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(arr)
        data = mem.read()
    write_atomic(block_path(item.collection, item.id, asset, row, col), data)
    west, south, east, north = array_bounds(h, w, t)
    minx, miny, maxx, maxy = transform_bounds(g["crs"], "EPSG:4326", west, south, east, north)
    remember(item)
    db().execute(
        "INSERT OR REPLACE INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item.collection, item.id, asset, row, col, minx, miny, maxx, maxy, len(data)),
    )


def window(item, asset: str, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    """Rows r0..r0+h and cols c0..c0+w of the asset as float32 (count, h, w): stored blocks,
    the rest fetched. Pixels off the raster or at nodata are NaN."""
    g = grid(item, asset)
    b = g["block"]
    out = np.full((g["count"], h, w), np.nan, dtype=np.float32)
    rows = range(max(r0, 0) // b, (min(r0 + h, g["height"]) - 1) // b + 1)
    cols = range(max(c0, 0) // b, (min(c0 + w, g["width"]) - 1) // b + 1)
    todo = [(r, c) for r in rows for c in cols]
    arrs = list(pool().map(lambda rc: block(item, asset, g, *rc), todo))
    for (r, c), arr in zip(todo, arrs):
        a = arr.astype(np.float32)
        if g["nodata"] is not None:
            a[arr == g["nodata"]] = np.nan
        y0, x0 = r * b - r0, c * b - c0  # the block's corner in the window
        ys, xs = max(y0, 0), max(x0, 0)
        ye, xe = min(y0 + a.shape[1], h), min(x0 + a.shape[2], w)
        if ye > ys and xe > xs:
            out[:, ys:ye, xs:xe] = a[:, ys - y0 : ye - y0, xs - x0 : xe - x0]
    return out


def stored_point(item, name: str, lon: float, lat: float) -> float | None:
    row = (
        db()
        .execute(
            "SELECT value FROM points WHERE collection = ? AND item = ? AND name = ? AND lon = ? "
            "AND lat = ?",
            (item.collection, item.id, name, round(lon, 6), round(lat, 6)),
        )
        .fetchone()
    )
    if row is None:
        return None
    return float("nan") if row[0] is None else row[0]  # SQLite keeps NaN as NULL


def save_points(item, values: dict[str, float], lon: float, lat: float) -> None:
    remember(item)
    db().executemany(
        "INSERT OR REPLACE INTO points VALUES (?, ?, ?, ?, ?, ?)",
        [(item.collection, item.id, k, round(lon, 6), round(lat, 6), v) for k, v in values.items()],
    )


# ---- listing and pruning ----


def where(collection=None, bbox=None, period=None, item=None, table="b") -> tuple[str, list]:
    sql, args = ["1"], []
    if collection:
        sql.append("i.collection = ?")
        args.append(collection)
    if item:
        sql.append("i.item = ?")
        args.append(item)
    if period:
        sql.append("substr(i.datetime, 1, 10) BETWEEN ? AND ?")
        args += list(period)
    if bbox:
        x0, x1, y0, y1 = ("b.minx", "b.maxx", "b.miny", "b.maxy") if table == "b" else (
            "p.lon", "p.lon", "p.lat", "p.lat")  # fmt: skip
        sql.append(f"{x1} >= ? AND {x0} <= ? AND {y1} >= ? AND {y0} <= ?")
        args += [bbox[0], bbox[2], bbox[1], bbox[3]]
    return " AND ".join(sql), args


def ls(**f) -> list[tuple]:
    w, args = where(**f)
    return (
        db()
        .execute(
            f"""SELECT i.collection, i.item, substr(i.datetime, 1, 10),
                   group_concat(DISTINCT b.asset), count(*), sum(b.bytes)
            FROM blocks b JOIN items i USING (collection, item) WHERE {w}
            GROUP BY i.collection, i.item ORDER BY i.collection, i.datetime""",
            args,
        )
        .fetchall()
    )


def usage() -> list[tuple]:
    """One row for each collection and 1-degree cell: region, period, blocks and bytes,
    largest first."""
    return (
        db()
        .execute(
            """SELECT i.collection,
                  CAST(b.minx + 180 AS int) - 180 AS lon_cell,
                  CAST(b.miny + 90 AS int) - 90 AS lat_cell,
                  min(b.minx), min(b.miny), max(b.maxx), max(b.maxy),
                  substr(min(i.datetime), 1, 10), substr(max(i.datetime), 1, 10),
                  count(*), sum(b.bytes)
           FROM blocks b JOIN items i USING (collection, item)
           GROUP BY i.collection, lon_cell, lat_cell
           ORDER BY sum(b.bytes) DESC"""
        )
        .fetchall()
    )


def total_bytes() -> int:
    return db().execute("SELECT coalesce(sum(bytes), 0) FROM blocks").fetchone()[0]


def n_points() -> int:
    return db().execute("SELECT count(*) FROM points").fetchone()[0]


def prune(dry_run: bool = False, **f) -> tuple[int, int, int]:
    """Delete the blocks and points that match; (blocks, points, bytes)."""
    w, args = where(**f)
    blocks = (
        db()
        .execute(
            "SELECT b.collection, b.item, b.asset, b.row, b.col, b.bytes "
            f"FROM blocks b JOIN items i USING (collection, item) WHERE {w}",
            args,
        )
        .fetchall()
    )
    pw, pargs = where(**f, table="p")
    points = (
        db()
        .execute(
            "SELECT p.collection, p.item, p.name, p.lon, p.lat FROM points p "
            "JOIN items i USING (collection, item) "
            f"WHERE {pw}",
            pargs,
        )
        .fetchall()
    )
    if not dry_run:
        con = db()
        con.execute("BEGIN")
        con.executemany(
            "DELETE FROM blocks WHERE collection = ? AND item = ? AND asset = ? AND row = ? "
            "AND col = ?",
            [b[:5] for b in blocks],
        )
        con.executemany(
            "DELETE FROM points WHERE collection = ? AND item = ? AND name = ? AND lon = ? "
            "AND lat = ?",
            points,
        )
        con.execute("COMMIT")
        for coll, item, asset, r, c, _ in blocks:
            block_path(coll, item, asset, r, c).unlink(missing_ok=True)
    return len(blocks), len(points), sum(b[5] for b in blocks)
