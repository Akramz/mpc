"""The Planetary Computer catalog as one short record per collection: a snapshot ships in the
package, and `mpc catalog --refresh` writes a newer local copy to `root()/catalog.json` (only
changed collections are rebuilt). Nothing refreshes on its own."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs

import httpx

from mpc.expr import names
from mpc.http import API, STAC, cached_json, root, write_atomic

SNAPSHOT = Path(__file__).with_name("catalog.json")
COG = "profile=cloud-optimized"
SKIP_ROLES = {"thumbnail", "overview", "metadata"}


def local() -> Path:
    return root() / "catalog.json"


def load() -> dict:
    """The newer of the local copy and the snapshot."""
    cats = [json.loads(p.read_text()) for p in (local(), SNAPSHOT) if p.exists()]
    if not cats:
        return {"updated": "1970-01-01T00:00:00Z", "collections": {}}
    return max(cats, key=stamp)


def stamp(cat: dict) -> float:
    return datetime.fromisoformat(cat["updated"]).timestamp()


def refresh(old: dict | None = None, out: Path | None = None, log=None) -> dict:
    """Rebuild from the live collections list; render presets are fetched again only for
    collections whose record changed. Writes `out` (the local copy by default)."""
    cols = live_collections()
    prev = (old or {}).get("collections", {})
    recs, todo = {}, []
    for c in cols:
        h = digest(c)
        if c["id"] in prev and prev[c["id"]].get("hash") == h:
            recs[c["id"]] = prev[c["id"]]
        else:
            todo.append(c)
    with ThreadPoolExecutor(8) as ex:
        for c, p in zip(todo, ex.map(lambda c: presets(c["id"]), todo)):
            recs[c["id"]] = compact(c, p)
    cat = {
        "updated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collections": dict(sorted(recs.items())),
    }
    write_atomic(out or local(), json.dumps(cat, separators=(",", ":")).encode())
    if log:
        added = sorted(set(recs) - set(prev))
        removed = sorted(set(prev) - set(recs))
        changed = sorted(c["id"] for c in todo if c["id"] in prev)
        for what, ids in (("added", added), ("changed", changed), ("removed", removed)):
            if ids:
                log(f"{what}: {', '.join(ids)}")
        log(f"{len(recs)} collections, {len(todo)} rebuilt")
    return cat


def live_collections() -> list[dict]:
    out, url = [], f"{STAC}/collections"
    while url:
        page = cached_json("GET", url, ttl=0)
        out += page["collections"]
        url = next((ln["href"] for ln in page.get("links", []) if ln.get("rel") == "next"), None)
    return out


def digest(c: dict) -> str:
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def presets(cid: str) -> dict[str, list[str]]:
    """MPC's render presets as mpc expressions: {"Natural color": ["B04", "B03", "B02"]}."""
    try:
        info = cached_json("GET", f"{API}/data/v1/mosaic/info", params={"collection": cid}, ttl=0)
    except httpx.HTTPError:
        return {}
    out = {}
    for r in info.get("renderOptions") or []:
        exprs = preset_exprs(r.get("options") or "")
        if exprs:
            out[r["name"]] = exprs
    return out


def preset_exprs(options: str) -> list[str]:
    q = parse_qs(options.replace("+", "%2B"))  # a "+" in an expression is a plus, not a space
    if "expression" in q:
        return [q["expression"][0].replace(" ", "")]
    assets = q.get("assets", [])
    bidx = dict(b.split("|", 1) for b in q.get("asset_bidx", []) if "|" in b)
    out = []
    for a in assets:
        if a in bidx:
            out += [f"{a}_b{i}" for i in bidx[a].split(",")]
        else:
            out.append(a)
    return out if len(out) in (1, 3) else []


def compact(c: dict, presets: dict) -> dict:
    gsd = (c.get("summaries") or {}).get("gsd")
    default_res = gsd[0] if isinstance(gsd, list) and len(gsd) == 1 else None
    assets = {}
    for key, a in (c.get("item_assets") or {}).items():
        roles = set(a.get("roles") or ["data"])
        if COG not in a.get("type", "") or roles <= SKIP_ROLES:
            continue
        eo, rb = a.get("eo:bands") or [], a.get("raster:bands") or []
        n = max(len(eo), len(rb), 1)
        for i in range(n):
            e = eo[i] if i < len(eo) else {}
            r = rb[i] if i < len(rb) else {}
            rec = {
                "about": about(a, e, n),
                "res": r.get("spatial_resolution") or a.get("gsd") or default_res,
                "dtype": r.get("data_type"),
                "scale": r.get("scale"),
                "offset": r.get("offset"),
                "nodata": r.get("nodata"),
                "unit": r.get("unit"),
                "classes": classes(a.get("file:values") or a.get("classification:classes")),
            }
            assets[key if n == 1 else f"{key}_b{i + 1}"] = {
                k: v for k, v in rec.items() if v is not None
            }
    (start, end), bbox = c["extent"]["temporal"]["interval"][0], c["extent"]["spatial"]["bbox"][0]
    return {
        "title": c.get("title", c["id"]),
        "about": c.get("msft:short_description") or (c.get("description") or "")[:240],
        "keywords": c.get("keywords") or [],
        "readable": bool(assets),
        "assets": assets,
        "presets": {k: v for k, v in presets.items() if all(names_ok(x, assets) for x in v)},
        "time": [start, end],
        "bbox": bbox,
        "license": c.get("license"),
        "hash": digest(c),
    }


def about(a: dict, e: dict, n: int) -> str:
    text = e.get("description") or (e.get("name") if n > 1 else None) or a.get("title") or ""
    if n > 1 and a.get("title") and text != a["title"]:
        text = f"{a['title']}: {text}"
    if e.get("common_name") and e["common_name"].lower() not in text.lower():
        text = f"{text} ({e['common_name']})".strip()
    if e.get("center_wavelength"):
        text += f", {round(e['center_wavelength'] * 1000)} nm"
    return text


def classes(values) -> dict[str, str] | None:
    """{"5": "Crops"} from file:values or classification:classes."""
    out = {}
    for v in values or []:
        if "values" in v:
            for x in v["values"]:
                out[str(x)] = v.get("summary", "")
        elif "value" in v:
            out[str(v["value"])] = v.get("description") or v.get("name", "")
    return out or None


def names_ok(expr: str, assets: dict) -> bool:
    return names(expr, assets) <= set(assets)


def get(cid: str, cat: dict | None = None) -> dict:
    cols = (cat or load())["collections"]
    if cid not in cols:
        raise ValueError(
            f"no collection {cid!r}; mpc catalog <words> finds collections, "
            "mpc catalog --refresh updates the list"
        )
    return {"id": cid, **cols[cid]}


def search(words: str, cat: dict | None = None) -> list[str]:
    """Collection ids ranked by matches in id, title, keywords and description."""
    cols = (cat or load())["collections"]
    terms = [w.lower() for w in words.split() if w]
    scored = []
    for cid, c in cols.items():
        kw = " ".join(c["keywords"]).lower()
        s = sum(
            3 * (t in cid) + 2 * (t in c["title"].lower()) + (t in kw) + (t in c["about"].lower())
            for t in terms
        )
        if s or not terms:
            scored.append((-s, not c["readable"], cid))
    return [cid for *_, cid in sorted(scored)]


def resolution(c: dict) -> float | None:
    res = [a["res"] for a in c["assets"].values() if a.get("res")]
    return min(res) if res else None


def years(c: dict) -> str:
    start, end = c["time"]
    a = start[:4] if start else "?"
    b = end[:4] if end else "now"
    return a if a == b else f"{a}-{b}"
