"""Navigation: cheap views of one scene from its COG overviews. A view is a path of quadrant
digits from the whole scene (zoom 0, at its coarsest overview); each digit halves the view and
reads the next finer overview, down to native pixels. Each view has an id, a hash of what it
shows, and lives in `root()/views/<id>.json`; `navigate --view` and `read --view --at` name it,
so agents that run at once never share a view."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
from rasterio.transform import Affine
from rasterio.warp import transform as warp

from mpc import catalog, cog, reader, sheet, stac, store
from mpc import expr as ex
from mpc.http import count, root, write_atomic
from mpc.units import physical

QUADRANTS = "1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right"
OFFSET = {"1": (0.0, 0.0), "2": (0.5, 0.0), "3": (0.0, 0.5), "4": (0.5, 0.5)}
MAX_PX = 1024  # agent CLIs scale larger images down
MIN_NATIVE_PX = 64


@dataclass(frozen=True)
class View:
    item: dict  # the scene's STAC record; navigation never leaves it
    place: tuple[float, float]  # the lon, lat the view started from
    path: str = ""  # quadrant digits joined by "/"; "" is the whole scene
    exprs: tuple[str, ...] = ()  # 1 or 3 expressions

    @property
    def collection(self) -> str:
        return self.item["collection"]


def ident(v: View) -> str:
    return hashlib.sha256(json.dumps(asdict(v), sort_keys=True).encode()).hexdigest()[:12]


def save(v: View) -> None:
    write_atomic(root() / "views" / f"{ident(v)}.json", json.dumps(asdict(v)).encode())


def load(vid: str) -> View:
    path = root() / "views" / f"{vid}.json"
    if not vid.isalnum() or not path.exists():
        raise ValueError(f"no view {vid!r}; mpc navigate prints the id of each view")
    d = json.loads(path.read_text())
    return View(**{**d, "place": tuple(d["place"]), "exprs": tuple(d["exprs"])})


def parse(path: str) -> str:
    digits = [p.strip() for p in str(path).split("/") if p.strip()]
    if any(p not in OFFSET for p in digits):
        raise ValueError(f'a path is digits 1-4 ({QUADRANTS}) joined by "/"; "" is the scene')
    return "/".join(digits)


def steps(v: View) -> list[str]:
    return [p for p in v.path.split("/") if p]


def grid(v: View) -> dict:
    """The grid of the view's first asset: crs, transform, width, height, overviews."""
    item = stac.Item(v.item)
    name = reader.check_exprs(catalog.get(v.collection), list(v.exprs))[0]
    return store.grid(item, reader.asset_of(item, name)[0], need=("overviews",))


def start(collection: str, lon: float, lat: float, date: str, exprs) -> View:
    """The scene on `date` at the place, found as `read --dates` finds it; zoom 0 is all of it."""
    coll = catalog.get(collection)
    reader.image_exprs(coll, exprs)
    item = stac.on(coll["id"], lon, lat, stac.day(date))
    return View(item.d, (lon, lat), "", tuple(exprs))


def box(v: View, g: dict) -> tuple[float, float, float]:
    """The view's top-left corner and side, in the scene's CRS."""
    t = Affine(*g["transform"])
    x0, y0, side = t.c, t.f, max(g["width"], g["height"]) * t.a
    for q in steps(v):
        dx, dy = OFFSET[q]
        x0, y0, side = x0 + dx * side, y0 - dy * side, side / 2
    return x0, y0, side


def native_px(v: View, g: dict) -> float:
    return max(g["width"], g["height"]) / 2 ** len(steps(v))


def pixels(v: View, g: dict) -> int:
    """Pixels across the view: the coarsest overview's size at every zoom until the view is at
    native resolution, then the native pixels it holds. Rounded down, so GDAL reads the
    overview `level` names."""
    coarsest = max(g["width"], g["height"]) / max(g["overviews"] or [1])
    return max(1, min(int(min(native_px(v, g), coarsest)), MAX_PX))


def level(v: View, g: dict, px: int) -> int:
    """The overview factor GDAL reads for the view; 1 is native."""
    ratio = native_px(v, g) / px
    return max([f for f in g["overviews"] if f <= ratio], default=1)


def lonlat(v: View, x: float, y: float) -> tuple[float, float]:
    """The WGS84 point `x`, `y` percent from the left and top of the view."""
    if not (0 <= x <= 100 and 0 <= y <= 100):
        raise ValueError("--at is percent from the left and top of the view, 0 to 100")
    g = grid(v)
    x0, y0, side = box(v, g)
    xs, ys = warp(g["crs"], "EPSG:4326", [x0 + side * x / 100], [y0 - side * y / 100])
    return float(xs[0]), float(ys[0])


def spot(v: View, g: dict, px: int) -> tuple[float, float] | None:
    """The place in pixels of the view; None outside it."""
    x0, y0, side = box(v, g)
    xs, ys = warp("EPSG:4326", g["crs"], [v.place[0]], [v.place[1]])
    x, y = (xs[0] - x0) / side * px, (y0 - ys[0]) / side * px
    return (x, y) if 0 <= x < px and 0 <= y < px else None


def caption(v: View, coll: dict, g: dict, px: int, mark: bool) -> str:
    item = stac.Item(v.item)
    side = box(v, g)[2]
    size = partial(reader.length, g["crs"])
    f = level(v, g, px)
    factors = [*sorted(g["overviews"], reverse=True), 1]
    levels = " ".join(f"[{k}x]" if k == f else f"{k}x" for k in factors)
    z = len(steps(v))
    where = "the whole scene" if z == 0 else f"zoom {z}"
    mark = "the start place is outlined in red" if mark else "the start place is outside this view"
    vid = ident(v)
    up = "/".join(steps(v)[:-1]) if z else None
    down = f"{v.path}/1-4" if v.path else "1-4"
    moves = f"--view {vid} --path {down} zooms in" + (
        f', --path "{up}" zooms out' if up is not None else ""
    )
    lines = [
        f"view {vid} · {coll['id']} · {item.label} · {item.id}",
        f'path "{v.path}" · {where} · {size(side)} across · {reader.crs_name(g["crs"])}',
        f"{px} x {px} px of {size(side / px)} · overview levels {levels} (1x is native "
        f"{size(reader.res_of(g))})"
        + ("" if g["overviews"] else "; no overviews: views read native pixels"),
        f"{reader.how(coll, v.exprs)}; quadrants {QUADRANTS}; ticks every 10 %; {mark}",
        f"next: {moves}; mpc read <collection> --view {vid} --at X,Y --dates ... reads a spot",
    ]
    return "\n".join(lines)


def show(v: View) -> tuple[Path, str]:
    """The view's PNG and caption; a view seen before costs no request."""
    g = grid(v)
    if native_px(v, g) < MIN_NATIVE_PX:
        raise ValueError(
            f'path "{v.path}" is under {MIN_NATIVE_PX} native pixels across: '
            "use mpc read --view ID --at X,Y --buffer METERS"
        )
    png = root() / "views" / f"{ident(v)}.png"
    txt = png.with_suffix(".txt")
    if png.exists() and txt.exists():
        count("hits")
        return png, txt.read_text()
    coll = catalog.get(v.collection)
    names = reader.image_exprs(coll, v.exprs)
    item = stac.Item(v.item)
    x0, y0, side = box(v, g)
    px = pixels(v, g)
    keys = sorted({reader.asset_of(item, n)[0] for n in names})
    with ThreadPoolExecutor(8) as pool:
        got = pool.map(lambda k: cog.overview(item, k, (x0, y0 - side, x0 + side, y0), px), keys)
        arrs = dict(zip(keys, got))
    count("requests", len(keys))
    bands = {}
    for n in names:
        k, b = reader.asset_of(item, n)
        bands[n] = physical(coll, item, n, arrs[k][b - 1])
    planes = np.stack([ex.evaluate(e, bands) for e in v.exprs])[None]
    place = spot(v, g, px)
    write_atomic(png, sheet.marked(reader.colored(coll, v.exprs, planes)[0], place))
    text = caption(v, coll, g, px, place is not None)
    write_atomic(txt, text.encode())
    return png, text
