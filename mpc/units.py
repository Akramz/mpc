"""Units the catalog does not state, by collection: one place for the reads and `mpc info`."""

from __future__ import annotations

import re

import numpy as np

S2_BANDS = re.compile(r"B\d\d|B8A")
S2_SCALE, S2_OFFSET, S2_BASELINE = 1e-4, -0.1, "04.00"
NOTES = {
    "sentinel-2-l2a": (
        "every asset: nodata 0 (COGs since 2026-04-17 carry no nodata tag, MPC #513); "
        f"B01-B12, B8A: reflectance = DN x {S2_SCALE:g}, {S2_OFFSET:+g} from processing "
        f"baseline {S2_BASELINE}"
    ),
}


def units(coll: dict, item, name: str) -> dict:
    a = coll["assets"][name]
    if coll["id"] == "sentinel-2-l2a":
        a = {**a, "nodata": 0}
        if S2_BANDS.fullmatch(name):
            new = item.props.get("s2:processing_baseline", "00.00") >= S2_BASELINE
            a.update(scale=S2_SCALE, offset=S2_OFFSET if new else 0.0, unit="reflectance")
    return a


def physical(coll: dict, item, name: str, raw) -> np.ndarray:
    """Raw values as float32 in the asset's units; nodata is NaN."""
    a = units(coll, item, name)
    x = np.asarray(raw, dtype=np.float32)
    out = x * np.float32(a.get("scale", 1)) + np.float32(a.get("offset", 0))
    if a.get("nodata") is not None:
        out = np.where(x == a["nodata"], np.float32(np.nan), out)
    return out
