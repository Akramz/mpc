"""`mpc`: catalog, info, search, navigate, read, store and install-skill."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from mpc import catalog, store
from mpc.http import STATS, root
from mpc.units import NOTES

AGENTS = ("~/.claude", "~/.copilot", "~/.codex")
STORE_WARN_BYTES = 10 * 2**30


def location(s: str) -> tuple[float, float]:
    try:
        lon, lat = (float(x) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("give lon,lat, e.g. 34.39767,0.48864") from None
    return lon, lat


def percent(s: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("give X,Y in percent, e.g. 62,35") from None
    return x, y


def bbox(s: str) -> tuple[float, ...]:
    try:
        b = tuple(float(x) for x in s.split(","))
    except ValueError:
        b = ()
    if len(b) != 4:
        raise argparse.ArgumentTypeError("give west,south,east,north")
    return b


def size(n: float | None) -> str:
    n = n or 0
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            break
        n /= 1024
    return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"


def cmd_catalog(a) -> None:
    cat = catalog.load()
    if a.refresh:
        cat = catalog.refresh(cat, log=print)
    ids = catalog.search(" ".join(a.words), cat)[: None if a.all else 15]
    cols = cat["collections"]
    w = max([len(i) for i in ids] + [2])
    print(f"{'id':<{w}}  {'title':<42}  {'years':<9}  res")
    for cid in ids:
        c = cols[cid]
        res = catalog.resolution(c)
        mark = "" if c["readable"] else "  (not readable)"
        print(
            f"{cid:<{w}}  {c['title'][:42]:<42}  {catalog.years(c):<9}  "
            f"{f'{res:g} m' if res else '-'}{mark}"
        )
    print(f"catalog {cat['updated'][:10]} · {len(cols)} collections", file=sys.stderr)


def cmd_info(a) -> None:
    c = catalog.get(a.collection)
    print(f"{c['id']}: {c['title']}")
    print(c["about"])
    print(f"time {catalog.years(c)} · license {c['license']}")
    if not c["readable"]:
        print("not readable by mpc: no COG assets")
        return
    w = max(len(n) for n in c["assets"])
    print(f"\n{'asset':<{w}}  {'res':>6}  {'dtype':<8}  units / about")
    for n, x in c["assets"].items():
        res = f"{x['res']:g} m" if x.get("res") else "-"
        extra = []
        if x.get("scale") is not None or x.get("offset") is not None:
            extra.append(f"value = DN x {x.get('scale', 1):g} + {x.get('offset', 0):g}")
        if x.get("nodata") is not None:
            extra.append(f"nodata {x['nodata']:g}")
        if x.get("unit"):
            extra.append(x["unit"])
        tail = f" [{'; '.join(extra)}]" if extra else ""
        print(f"{n:<{w}}  {res:>6}  {x.get('dtype') or '-':<8}  {x['about']}{tail}")
        for v, label in (x.get("classes") or {}).items():
            print(f"{'':<{w}}            {v:>4} {label}")
    if c["id"] in NOTES:
        print(f"\nmpc applies: {NOTES[c['id']]}")
    if c["presets"]:
        print("\npresets (use as --expr):")
        for k, v in c["presets"].items():
            print(f"  {k}: {' '.join(v)}")


def cell(v) -> str:
    if v is None:
        return "-"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def aligned(rows: list[list[str]]) -> str:
    ws = [max(len(x) for x in col) for col in zip(*rows)]
    return "\n".join("  ".join(x.ljust(w) for x, w in zip(r, ws)).rstrip() for r in rows)


def cmd_search(a) -> None:
    from mpc import stac

    lon, lat = a.location
    found = stac.scenes(catalog.get(a.collection)["id"], lon, lat, a.period)
    if a.json:
        print(json.dumps([{"date": i.label, "item": i.id, **i.props} for i in found], indent=1))
        return
    if not found:
        print(f"no {a.collection} scene at {lon}, {lat} in {a.period[0]}..{a.period[1]}")
        return
    times = {"datetime", "start_datetime", "end_datetime"}
    keys = [
        k for k in sorted({k for i in found for k in i.props})
        if k not in times and not any(isinstance(i.props.get(k), list | dict) for i in found)
    ]  # fmt: skip
    same = [k for k in keys if len({cell(i.props.get(k)) for i in found}) == 1]
    if same:
        print(aligned([[k, cell(found[0].props.get(k))] for k in same]) + "\n")
    cols = [k for k in keys if k not in same]
    rows = [[i.label, *(cell(i.props.get(k)) for k in cols), i.id] for i in found]
    print(aligned([["date", *cols, "item"], *rows]))
    print(
        f"{len(found)} scene{'s' * (len(found) != 1)} · {len(cols)} properties differ · "
        "--json gives each in full",
        file=sys.stderr,
    )


def cmd_navigate(a) -> None:
    from dataclasses import replace

    from mpc import nav

    exprs = tuple(a.expr or ())
    if a.view:
        if a.collection or a.location or a.date:
            raise ValueError("give --view ID, or <collection> --location --date for a new view")
        v = nav.load(a.view)
        v = replace(v, exprs=exprs or v.exprs)
    elif a.collection and a.location and a.date:
        v = nav.start(a.collection, *a.location, a.date, exprs)
    else:
        raise ValueError(
            "give <collection> --location LON,LAT --date YYYY-MM-DD for a new view "
            "(mpc search lists the dates), or --view ID"
        )
    if a.path is not None:
        v = replace(v, path=nav.parse(a.path))
    png, text = nav.show(v)
    nav.save(v)
    if a.json:
        out = {"view": nav.ident(v), "image": str(png), "caption": text, "path": v.path}
        print(json.dumps(out, indent=1))
        return
    print(text)
    print(png)


def cmd_read(a) -> None:
    from mpc.reader import Read, caption, crs_name, length, records, run, table

    exprs = tuple(a.expr)
    spot = None
    if a.at:
        from mpc import nav

        if not a.view:
            raise ValueError(
                "--at X,Y is a spot in a view: give --view ID (mpc navigate prints it)"
            )
        v = nav.load(a.view)
        lon, lat = nav.lonlat(v, *a.at)
        item = v.item["id"]
        spot = {"x": a.at[0], "y": a.at[1], "view": a.view, "path": v.path, "item": item,
                "lon": lon, "lat": lat}  # fmt: skip
    elif a.view:
        raise ValueError("--view goes with --at X,Y")
    else:
        lon, lat = a.location
    req = Read(
        a.collection, lon, lat, exprs, tuple(a.dates or ()),
        tuple(a.period) if a.period else None, a.buffer, a.save,
    )  # fmt: skip
    res = run(req)
    if a.json:
        out = {"collection": res.collection["id"], "items": records(res)}
        if spot:
            out["at"] = spot
        if res.view:
            out |= {"image": str(res.view), "caption": caption(res)}
        if res.saved:
            out["saved"] = str(res.saved)
        print(json.dumps(out, indent=1))
        return
    if spot:
        print(
            f'x {spot["x"]:g}, y {spot["y"]:g} of view {a.view} (path "{spot["path"]}", '
            f"{spot['item']}) "
            f"is lon {lon:.5f}, lat {lat:.5f}."
        )
    if res.values is not None:
        print(table(res))
    if res.view:
        print(caption(res))
        print(res.view)
    if res.saved:
        t, e, h, w = res.planes.shape
        print(
            f"saved {res.saved} · {w} x {h} px · {crs_name(res.crs)} · "
            f"{length(res.crs, abs(res.transform.a))} · {t} dates x {e} bands"
        )


def cmd_store(a) -> None:
    if a.what == "ls":
        rows = store.ls(collection=a.collection, bbox=a.bbox)
        print(f"{'collection':<20} {'date':<10}  {'item':<60} {'assets':<16} blocks  size")
        for coll, item, d, assets, n, b in rows:
            print(f"{coll:<20} {d:<10}  {item:<60} {assets:<16} {n:>6}  {size(b)}")
    elif a.what == "du":
        rows = store.usage()
        head = f"{'collection':<20} {'region (w,s,e,n)':<28} {'period':<23} {'blocks':>6}"
        print(f"{'#':>2}  {head}  size")
        for k, (coll, _, _, w, s, e, n, d0, d1, nb, b) in enumerate(rows, 1):
            region = f"{w:.2f},{s:.2f},{e:.2f},{n:.2f}"
            period = d0 if d0 == d1 else f"{d0}/{d1}"
            print(f"{k:>2}  {coll:<20} {region:<28} {period:<23} {nb:>6}  {size(b)}")
        print(f"{root() / 'store'} · {size(store.total_bytes())} · {store.n_points()} points")
    else:
        from mpc import stac

        period = tuple(d.isoformat() for d in stac.span(a.period)) if a.period else None
        f = dict(collection=a.collection, bbox=a.bbox, item=a.item, period=period)
        if not any(f.values()) and not a.everything:
            raise ValueError("name what to prune: --collection, --bbox, --period, --item or --all")
        nb, npts, b = store.prune(dry_run=a.dry_run, **f)
        verb = "would remove" if a.dry_run else "removed"
        print(f"{verb} {nb} blocks and {npts} points · {size(b)}")


def cmd_install_skill(a) -> None:
    src = Path(__file__).with_name("skill")
    done = 0
    for agent in AGENTS:
        home = Path(agent).expanduser()
        if not home.is_dir():
            continue
        dst = home / "skills" / "mpc"
        if dst.exists() and not dst.is_symlink():
            print(f"skip {dst}: a folder that mpc did not make")
            continue
        if not a.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src, target_is_directory=True)
        print(f"✓ {dst} -> {src}")
        done += 1
    if not done:
        print("no agent folder found (~/.claude, ~/.copilot, ~/.codex)")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mpc", description="Read the Microsoft Planetary Computer.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("catalog", help="find collections by words")
    c.add_argument("words", nargs="*")
    c.add_argument("--all", action="store_true", help="list every match, not the first 15")
    c.add_argument("--refresh", action="store_true", help="rebuild from the live catalog")
    c.set_defaults(fn=cmd_catalog)

    i = sub.add_parser("info", help="a collection's assets, units, classes and presets")
    i.add_argument("collection")
    i.set_defaults(fn=cmd_info)

    f = sub.add_parser("search", help="the scenes at a location in a period, with their properties")
    f.add_argument("collection")
    f.add_argument("--location", type=location, required=True, metavar="LON,LAT")
    f.add_argument("--period", nargs=2, required=True, metavar=("START", "END"))
    f.add_argument("--json", action="store_true", help="every property of each scene")
    f.set_defaults(fn=cmd_search)

    n = sub.add_parser("navigate", help="cheap views of one scene, from its COG overviews")
    n.add_argument("collection", nargs="?", help="with --location and --date: a new view")
    n.add_argument("--location", type=location, metavar="LON,LAT")
    n.add_argument("--date", metavar="YYYY-MM-DD", help="the scene's date, as mpc search lists it")
    n.add_argument("--view", metavar="ID", help="a view navigate printed")
    n.add_argument(
        "--path", metavar="2/4/1", help='quadrant digits from the whole scene; "" is all of it'
    )
    n.add_argument(
        "--expr",
        nargs="+",
        metavar="EXPR",
        help="one (gray) or three (RGB) asset names or expressions",
    )
    n.add_argument("--json", action="store_true")
    n.set_defaults(fn=cmd_navigate)

    r = sub.add_parser("read", help="values or an image at a location")
    r.add_argument("collection")
    place = r.add_mutually_exclusive_group(required=True)
    place.add_argument("--location", type=location, metavar="LON,LAT")
    place.add_argument(
        "--at", type=percent, metavar="X,Y", help="percent from the left and top of --view"
    )
    r.add_argument("--view", metavar="ID", help="with --at: the view navigate printed")
    r.add_argument(
        "--expr", nargs="+", required=True, metavar="EXPR",
        help='asset names or expressions over them, e.g. B04 "(B08-B04)/(B08+B04)"',
    )  # fmt: skip
    when = r.add_mutually_exclusive_group(required=True)
    when.add_argument("--dates", nargs="+", metavar="YYYY-MM-DD", help="the scene on each day")
    when.add_argument(
        "--period", nargs=2, metavar=("START", "END"), help="every scene, ends included"
    )
    r.add_argument("--buffer", type=float, metavar="METERS", help="an image this far each side")
    r.add_argument(
        "--save", metavar="FILE.tif", help="write the native window stack (needs --buffer)"
    )
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_read)

    s = sub.add_parser("store", help="what the local store holds: ls, du, prune")
    s.add_argument("what", choices=("ls", "du", "prune"))
    s.add_argument("--collection")
    s.add_argument("--bbox", type=bbox, metavar="W,S,E,N")
    s.add_argument("--period", nargs=2, metavar=("START", "END"))
    s.add_argument("--item")
    s.add_argument("--all", dest="everything", action="store_true", help="prune: everything")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_store)

    k = sub.add_parser("install-skill", help="link the mpc skill for Claude, Copilot and Codex")
    k.add_argument("--dry-run", action="store_true")
    k.set_defaults(fn=cmd_install_skill)
    return p


def joined(argv: list[str]) -> list[str]:
    """`--location -76.05,38.77`: argparse reads a leading minus as an option; join it."""
    out, it = [], iter(argv)
    for x in it:
        if x in ("--location", "--bbox"):
            nxt = next(it, None)
            out.append(f"{x}={nxt}" if nxt is not None else x)
        else:
            out.append(x)
    return out


def main(argv: list[str] | None = None) -> None:
    a = parser().parse_args(joined(sys.argv[1:] if argv is None else argv))
    t = time.time()
    try:
        a.fn(a)
    except (ValueError, LookupError, OSError, httpx.HTTPError) as e:  # OSError: rasterio I/O too
        print(f"error: {str(e).splitlines()[0]}", file=sys.stderr)
        sys.exit(1)
    if a.cmd in ("read", "navigate"):
        sys.stdout.flush()
        print(
            f"mpc: {STATS['requests']} requests · store: {STATS['hits']} hits · "
            f"{time.time() - t:.1f} s",
            file=sys.stderr,
        )
        if (held := store.total_bytes()) > STORE_WARN_BYTES:
            print(
                f"mpc: the store holds {size(held)}; mpc store du shows what uses it",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
