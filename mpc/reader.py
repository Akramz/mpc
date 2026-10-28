"""One read: a place, asset names or expressions over them, dates or a period, and a buffer.
Without a buffer each item gives the native pixel at the point (data API, stored as points);
with one, a native window from the COG blocks (stored as blocks), shown as one PNG sheet and,
with `save`, written as one GeoTIFF stack."""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol
from rasterio.warp import transform as warp
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from mpc import catalog, sheet, stac, store
from mpc import expr as ex
from mpc.http import API, count, fetch, root, write_atomic
from mpc.units import physical

MAX_VIEW_PX = 1025  # pixels on a side of a viewed window
MAX_SAVE_PX = 10001
MAX_VIEW_DATES = 12
MAX_STACK_BYTES = 4 * 2**30  # the planes of one read, held in memory


@dataclass(frozen=True)
class Read:
    collection: str
    lon: float
    lat: float
    exprs: tuple[str, ...]  # asset names or expressions; an image takes 1 (gray) or 3 (RGB)
    dates: tuple[str, ...] = ()  # each gives the scene on that day
    period: tuple[str, str] | None = None  # start, end: every scene, both included
    buffer_m: float | None = None  # None: the pixel at the point
    save: str | None = None  # a GeoTIFF path for the native stack


@dataclass
class Result:
    read: Read
    collection: dict
    exprs: list[str]
    items: list[stac.Item]
    values: np.ndarray | None = None  # (T, E) without a buffer
    planes: np.ndarray | None = None  # (T, E, H, W) with one
    crs: str | None = None
    transform: Affine | None = None
    notes: list[str] = field(default_factory=list)
    view: Path | None = None
    saved: Path | None = None

    @property
    def dates(self) -> list[str]:
        return [i.label for i in self.items]


def asset_of(item, name: str) -> tuple[str, int]:
    if name in item.d["assets"]:
        return name, 1
    key, _, b = name.rpartition("_b")
    if key in item.d["assets"] and b.isdigit():
        return key, int(b)
    raise ValueError(f"item {item.id} has no asset {name!r}")


def pixel(g: dict, lon: float, lat: float) -> tuple[int, int] | None:
    """The (row, col) that holds the point in the grid; None off the raster."""
    xs, ys = warp("EPSG:4326", g["crs"], [lon], [lat])
    r, c = rowcol(Affine(*g["transform"]), xs[0], ys[0])
    return (int(r), int(c)) if 0 <= r < g["height"] and 0 <= c < g["width"] else None


# ---- points ----


def point_values(item, names, lon, lat) -> dict[str, float]:
    """Raw values at the point: stored points, then stored blocks, then one data API call."""
    raw = {}
    for n in names:
        v = store.stored_point(item, n, lon, lat)
        if v is None:
            v = from_blocks(item, n, lon, lat)
        if v is not None:
            raw[n] = v
            count("hits")
    missing = [n for n in names if n not in raw]
    if missing:
        keys = sorted({asset_of(item, n)[0] for n in missing})
        params = [("collection", item.collection), ("item", item.id)]
        params += [("assets", k) for k in keys]
        try:
            url = f"{API}/data/v1/item/point/{lon},{lat}"
            d = fetch("GET", url, params=params, tries=2).json()
            got = dict(zip(d["band_names"], d["values"]))
        except httpx.HTTPStatusError:
            got = None  # the API cannot serve this point (or it is off the raster): ask the COG
        new = {}
        for n in missing:
            k, b = asset_of(item, n)
            v = cog_pixel(item, n, lon, lat) if got is None else got.get(f"{k}_b{b}")
            new[n] = math.nan if v is None else float(v)
        store.save_points(item, new, lon, lat)
        raw |= new
    return raw


def from_blocks(item, name, lon, lat) -> float | None:
    key, b = asset_of(item, name)
    g = store.cached_grid(item, key)
    rc = pixel(g, lon, lat) if g else None
    if rc is None:
        return None
    n = g["block"]
    path = store.stored_block(item, key, rc[0] // n, rc[1] // n)
    return None if path is None else float(store.read_block(path)[b - 1, rc[0] % n, rc[1] % n])


def cog_pixel(item, name, lon, lat) -> float:
    """The raw pixel from the COG block that holds it; NaN off the raster."""
    key, b = asset_of(item, name)
    rc = pixel(store.grid(item, key), lon, lat)
    return math.nan if rc is None else float(store.window(item, key, *rc, 1, 1)[b - 1, 0, 0])


# ---- windows ----


def res_of(g: dict) -> float:
    return abs(g["transform"][0])


def crs_name(crs: str) -> str:
    """EPSG:32636, or the PROJ string of a CRS without an EPSG code (MODIS sinusoidal)."""
    c = CRS.from_string(crs)
    if c.to_epsg():
        return f"EPSG:{c.to_epsg()}"
    return " ".join(x for x in c.to_proj4().split() if not x.startswith("+no_defs"))


def length(crs: str, x: float) -> str:
    """A length in the grid's units: degrees, or meters and kilometers."""
    if not CRS.from_string(crs).is_projected:
        return f"{x:.4g} deg"
    return f"{x / 1000:.1f} km" if x >= 1000 else f"{x:.3g} m"


def plan(grids: dict, lon, lat, buffer_m, limit, notes) -> dict:
    """Which rows and cols of each asset a window needs, and how each lands on the finest
    grid, `buffer_m` around the point's pixel, which sits at the center."""
    if len({g["crs"] for g in grids.values()}) > 1:
        raise ValueError(f"the assets {', '.join(grids)} are in different CRSs")
    fk = min(grids, key=lambda k: res_of(grids[k]))
    f, res = grids[fk], res_of(grids[fk])
    if f["transform"][1] or f["transform"][3]:
        raise ValueError(f"{fk} has a rotated grid; mpc reads north-up grids only")
    xs, ys = warp("EPSG:4326", f["crs"], [lon], [lat])
    r, c = (int(v) for v in rowcol(Affine(*f["transform"]), xs[0], ys[0]))
    # meters per pixel; on a grid in degrees, east-west at the point's latitude
    step = (
        res
        if CRS.from_string(f["crs"]).is_projected
        else res * 111_320 * math.cos(math.radians(lat))
    )
    half = max(0, round(buffer_m / step))
    n, r0, c0 = 2 * half + 1, r - half, c - half
    if n > limit:
        size = length(f["crs"], res)
        raise ValueError(f"the window is {n} pixels of {size} on a side; the limit is {limit}")
    reads, crops = {}, {}
    for k, g in grids.items():
        kf = res_of(g) / res
        dy = (f["transform"][5] - g["transform"][5]) / res
        dx = (g["transform"][2] - f["transform"][2]) / res
        if any(abs(v - round(v)) > 1e-6 for v in (kf, dx, dy)):
            raise ValueError(
                f"{k} ({length(g['crs'], res_of(g))}) does not share a pixel grid with {fk}"
            )
        kf, dx, dy = round(kf), round(dx), round(dy)
        R0, C0 = math.floor((r0 - dy) / kf), math.floor((c0 - dx) / kf)
        R1, C1 = math.ceil((r0 + n - dy) / kf), math.ceil((c0 + n - dx) / kf)
        reads[k] = (R0, C0, R1 - R0, C1 - C0)
        crops[k] = (kf, r0 - (dy + R0 * kf), c0 - (dx + C0 * kf))
        note = f"{k} is {length(g['crs'], res_of(g))}; each of its pixels repeats {kf} x {kf}"
        if kf > 1 and (note := f"{note} on the {length(g['crs'], res)} grid") not in notes:
            notes.append(note)
    t = window_transform(Window(c0, r0, n, n), Affine(*f["transform"]))
    return {"n": n, "crs": f["crs"], "transform": t, "reads": reads, "crops": crops}


def native(coll, items, names, lon, lat, buffer_m, limit, notes) -> list[tuple]:
    """(arrays, crs, transform) for each item: physical arrays of `names` on the finest grid
    among their assets; coarser assets repeat their pixels. Headers, then windows, of every
    (item, asset) run on one pool."""
    keys = [{nm: asset_of(i, nm) for nm in names} for i in items]
    jobs = sorted({(t, k) for t, ks in enumerate(keys) for k, _ in ks.values()})
    with ThreadPoolExecutor(16) as pool:
        grids = dict(zip(jobs, pool.map(lambda j: store.grid(items[j[0]], j[1]), jobs)))
        plans = [
            plan({k: grids[(t, k)] for k, _ in ks.values()}, lon, lat, buffer_m, limit, notes)
            for t, ks in enumerate(keys)
        ]
        size = sum(p["n"] ** 2 for p in plans) * len(names) * 4
        if size > MAX_STACK_BYTES:
            raise ValueError(f"the read needs {size / 2**30:.1f} GB; narrow the buffer or dates")
        wins = pool.map(
            lambda j: store.window(items[j[0]], j[1], *plans[j[0]]["reads"][j[1]]), jobs
        )
        wins = dict(zip(jobs, wins))
    out = []
    for t, (item, ks, p) in enumerate(zip(items, keys, plans)):
        arrays = {}
        for nm, (k, b) in ks.items():
            kf, oy, ox = p["crops"][k]
            a = wins[(t, k)][b - 1].repeat(kf, 0).repeat(kf, 1) if kf > 1 else wins[(t, k)][b - 1]
            arrays[nm] = physical(coll, item, nm, a[oy : oy + p["n"], ox : ox + p["n"]])
        out.append((arrays, p["crs"], p["transform"]))
    return out


# ---- the read ----


def check_exprs(coll: dict, exprs) -> list[str]:
    """The asset names `exprs` use; fails on an unreadable collection or an unknown name."""
    if not coll["readable"]:
        raise ValueError(f"{coll['id']} has no COG assets mpc can read")
    if not exprs:
        raise ValueError("give --expr; mpc info lists the assets and color presets")
    names = set()
    for e in exprs:
        used = ex.names(e, coll["assets"])
        if not used:
            raise ValueError(f"{e!r} names no asset")
        unknown = sorted(used - set(coll["assets"]))
        if unknown:
            raise ValueError(
                f"unknown asset {unknown[0]!r}; {coll['id']} has {', '.join(coll['assets'])}"
            )
        names |= used
    return sorted(names)


def image_exprs(coll: dict, exprs) -> list[str]:
    """check_exprs for an image: one expression (grayscale) or three (RGB)."""
    names = check_exprs(coll, exprs)
    if len(exprs) not in (1, 3):
        raise ValueError("an image takes one expression (grayscale) or three (RGB)")
    return names


def run(req: Read, cat: dict | None = None) -> Result:
    coll = catalog.get(req.collection, cat)
    if not (-180 <= req.lon <= 180 and -90 <= req.lat <= 90):
        raise ValueError(f"lon {req.lon}, lat {req.lat} is not a WGS84 location")
    image = req.buffer_m is not None
    if req.save and not image:
        raise ValueError("--save writes a window: give --buffer")
    exprs = list(req.exprs)
    names = (image_exprs if image and not req.save else check_exprs)(coll, exprs)
    lon, lat = req.lon, req.lat
    items = stac.items(coll["id"], lon, lat, req.dates, req.period)
    drawable = len(exprs) in (1, 3) and len(items) <= MAX_VIEW_DATES
    if image and not req.save and not drawable:
        raise ValueError(f"{len(items)} dates; an image shows at most {MAX_VIEW_DATES} (or --save)")
    res = Result(req, coll, exprs, items)
    if not image:
        rows = []
        for item in items:
            raw = point_values(item, names, lon, lat)
            bands = {n: np.array([physical(coll, item, n, raw[n])]) for n in names}
            rows.append([float(ex.evaluate(e, bands)[0]) for e in exprs])
        res.values = np.array(rows, dtype=np.float64).reshape(len(items), len(exprs))
        return res

    limit = MAX_SAVE_PX if req.save else MAX_VIEW_PX
    out = native(coll, items, names, lon, lat, req.buffer_m, limit, res.notes)
    res.planes = np.stack([np.stack([ex.evaluate(e, a) for e in exprs]) for a, _, _ in out])
    res.crs, res.transform = out[0][1], out[0][2]
    if any(t != res.transform or crs != res.crs for _, crs, t in out):
        if req.save:
            raise ValueError(
                "the dates come from different grids; --save needs one: save them apart"
            )
        res.notes.append("the dates come from different grids; each tile shows its own")
    if drawable:
        res.view = draw(res)
    if req.save:
        res.saved = save(res, Path(req.save))
    return res


# ---- outputs ----


def class_values(coll: dict, exprs) -> dict[str, str] | None:
    return coll["assets"].get(exprs[0], {}).get("classes") if len(exprs) == 1 else None


def colored(coll: dict, exprs, planes: np.ndarray) -> np.ndarray:
    """(T, E, H, W) planes to (T, H, W, 3) colors: class colors or one percentile stretch."""
    cls = class_values(coll, exprs)
    return sheet.classes(planes, [float(v) for v in cls]) if cls else sheet.stretch(planes)


def how(coll: dict, exprs, over: str = "") -> str:
    """How an image shows `exprs`, for its caption."""
    if legend := class_values(coll, exprs):
        return f"{exprs[0]} classes: " + ", ".join(
            f"{v} {name} = {sheet.color(i)[0]}" for i, (v, name) in enumerate(legend.items())
        )
    kind = f"RGB = {', '.join(exprs)}" if len(exprs) == 3 else f"grayscale = {exprs[0]}"
    return f"{kind}, 2..98 percentile stretch{over}"


def draw(res: Result) -> Path:
    rgb = colored(res.collection, res.exprs, res.planes)
    key = hashlib.sha256(json.dumps(asdict(res.read), sort_keys=True).encode()).hexdigest()[:12]
    path = root() / "views" / f"{key}.png"
    write_atomic(path, sheet.sheet(list(zip(res.dates, rgb))))
    return path


def caption(res: Result) -> str:
    r = res.read
    n = res.planes.shape[-1]
    px = abs(res.transform.a)
    text = (
        f"{res.collection['id']}, {n} x {n} pixels of {length(res.crs, px)} "
        f"({length(res.crs, n * px)} across), {crs_name(res.crs)}, "
        f"{how(res.collection, res.exprs, ' over all dates and bands')}; "
        f"the pixel at lon {r.lon:.5f}, lat {r.lat:.5f} is outlined in red."
    )
    if len(res.items) > 1:
        text += f" Dates, left to right, top to bottom: {', '.join(res.dates)}."
    return " ".join([text, *(f"{x[0].upper()}{x[1:]}." for x in res.notes)])


def table(res: Result) -> str:
    cls = {e: res.collection["assets"].get(e, {}).get("classes") for e in res.exprs}

    def fmt(e: str, v: float) -> str:
        if cls.get(e) and np.isfinite(v):
            return f"{v:g} {cls[e].get(str(int(v)), '')}".strip()
        if np.isfinite(v) and float(v).is_integer():
            return str(int(v))
        return ex.fmt_value(v)

    values = {e: res.values[:, j] for j, e in enumerate(res.exprs)}
    return ex.series_table(values, res.dates, fmt)


def records(res: Result) -> list[dict]:
    out = []
    for t, item in enumerate(res.items):
        rec = {"date": item.label, "item": item.id}
        if res.values is not None:
            for j, e in enumerate(res.exprs):
                v = res.values[t, j]
                rec[e] = float(v) if np.isfinite(v) else None
        out.append(rec)
    return out


def save(res: Result, path: Path) -> Path:
    t, e, h, w = res.planes.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", width=w, height=h, count=t * e, dtype="float32", crs=res.crs,
        transform=res.transform, nodata=np.nan, compress="deflate", predictor=3,
    )  # fmt: skip
    if h >= 256 and w >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        for i, (d, x) in enumerate(((d, x) for d in range(t) for x in range(e)), 1):
            dst.write(res.planes[d, x], i)
            dst.set_band_description(i, f"{res.dates[d]}_{res.exprs[x]}")
        dst.update_tags(
            times=json.dumps(res.dates),
            bands=json.dumps(res.exprs),
            items=json.dumps([i.id for i in res.items]),
            collection=res.collection["id"],
        )
    return path
