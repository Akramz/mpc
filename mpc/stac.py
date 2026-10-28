"""Scenes at a point and the date rules: a date gets the scene on that day (an acquisition that
UTC day, or an interval that holds it) or fails, a period every scene in it (both ends
included). Overlapping tiles give two items for one acquisition; each acquisition keeps the
first by id. `search`, `read` and `navigate` all see the same scenes."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time, timedelta

from mpc.http import HOUR, STAC, cached_json

CELL = 0.05  # degrees: the search area shared by nearby points


def when(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class Item:
    def __init__(self, d: dict):
        self.d, self.id, self.collection = d, d["id"], d["collection"]
        p = d["properties"]
        self.start = when(p.get("start_datetime")) or when(p.get("datetime"))
        self.end = when(p.get("end_datetime")) or self.start
        self.datetime = when(p.get("datetime")) or self.start
        # an interval ending at 00:00 covers the day before: an annual map ends on January 1
        midnight = self.end.time() == time() and self.end > self.start
        self.last = self.end - timedelta(microseconds=1) if midnight else self.end

    @property
    def props(self) -> dict:
        return self.d["properties"]

    @property
    def take(self) -> datetime:
        """The acquisition: the time to the minute."""
        return self.datetime.replace(second=0, microsecond=0)

    @property
    def label(self) -> str:
        if self.start.date() != self.last.date():
            return f"{self.start.date()}/{self.last.date()}"
        return self.datetime.date().isoformat()

    def href(self, asset: str) -> str:
        return self.d["assets"][asset]["href"]

    def __repr__(self) -> str:
        return f"Item({self.id})"


def day(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{s!r} is not a date; use YYYY-MM-DD") from None


def interval(a: date, b: date) -> str:
    return f"{a.isoformat()}T00:00:00Z/{b.isoformat()}T23:59:59Z"


def inside(rings, x: float, y: float) -> bool:
    """The even-odd rule over a polygon's rings."""
    odd = False
    for ring in rings:
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                odd = not odd
    return odd


def contains(geom: dict, lon: float, lat: float) -> bool:
    """Whether a GeoJSON Polygon or MultiPolygon holds the point. A ring that crosses the
    antimeridian (an edge spans more than 180 degrees) is unwrapped to 0..360 first."""
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom.get("coordinates", [])
    for rings in polys:
        edges = (x2 - x1 for ring in rings for (x1, _), (x2, _) in zip(ring, ring[1:]))
        if any(abs(d) > 180 for d in edges):
            rings = [[(x + 360 if x < 0 else x, y) for x, y in ring] for ring in rings]
            if inside(rings, lon, lat) or inside(rings, lon + 360, lat):
                return True
        elif inside(rings, lon, lat):
            return True
    return False


def search(coll: str, lon: float, lat: float, dt: str):
    """Items whose footprint holds the point. The search covers the CELL-degree cell around
    the point, so every point in that cell reuses one answer, cached for an hour."""
    x0, y0 = math.floor(lon / CELL) * CELL, math.floor(lat / CELL) * CELL
    body: dict = {
        "collections": [coll],
        "bbox": [round(x0, 6), round(y0, 6), round(x0 + CELL, 6), round(y0 + CELL, 6)],
        "limit": 1000,
        "datetime": dt,
    }
    out, url = [], f"{STAC}/search"
    while True:
        page = cached_json("POST", url, body=body, ttl=HOUR)
        out += page.get("features", [])
        nxt = next((ln for ln in page.get("links", []) if ln.get("rel") == "next"), None)
        if not nxt or not nxt.get("body"):
            break
        url, body = nxt["href"], {**body, **nxt["body"]}
    return [Item(f) for f in out if not f.get("geometry") or contains(f["geometry"], lon, lat)]


def acquisitions(found: list[Item]) -> list[list[Item]]:
    groups: dict[datetime, list[Item]] = {}
    for i in sorted(found, key=lambda i: (i.take, i.id)):
        groups.setdefault(i.take, []).append(i)
    return list(groups.values())


def span(period) -> tuple[date, date]:
    a, b = day(period[0]), day(period[1])
    if b < a:
        raise ValueError(f"the period ends ({b}) before it starts ({a})")
    return a, b


def between(coll, lon, lat, a: date, b: date) -> list[Item]:
    """Items at the point that touch days a..b."""
    start = datetime(a.year, a.month, a.day, tzinfo=UTC)
    return [i for i in search(coll, lon, lat, interval(a, b)) if i.last >= start]


def scenes(coll, lon, lat, period) -> list[Item]:
    """One item for each acquisition in the period, oldest first."""
    return [g[0] for g in acquisitions(between(coll, lon, lat, *span(period)))]


def on(coll, lon, lat, d: date) -> Item:
    """The scene on day `d`: its first acquisition that day."""
    for g in acquisitions(between(coll, lon, lat, d, d)):
        return g[0]
    raise LookupError(f"no {coll} scene at {lon}, {lat} on {d}; mpc search lists the scenes there")


def items(coll, lon, lat, dates=(), period=None) -> list[Item]:
    """The scenes a read shows, oldest first, each once."""
    if bool(dates) == bool(period):
        raise ValueError("give dates or a period (mpc search lists the scenes)")
    if period:
        if not (found := scenes(coll, lon, lat, period)):
            raise LookupError(f"no {coll} scene at {lon}, {lat} from {period[0]} to {period[1]}")
        return found
    days = [day(s) for s in dates]
    with ThreadPoolExecutor(8) as ex:
        picks = list(ex.map(lambda d: on(coll, lon, lat, d), days))
    return sorted({i.id: i for i in picks}.values(), key=lambda i: i.datetime)
