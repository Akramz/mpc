"""HTTP to the Planetary Computer: retries on 429 and 5xx, request and hit counts, and a JSON
cache of one file per request under `root()/http`, each read with a lifetime of its own."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from functools import cache
from pathlib import Path

import httpx

API = "https://planetarycomputer.microsoft.com/api"
STAC = f"{API}/stac/v1"
HOUR = 3600
STATS = {"requests": 0, "hits": 0}
_lock = threading.Lock()
_client: httpx.Client | None = None


def root() -> Path:
    return Path(os.environ.get("MPC_HOME") or "~/.cache/mpc").expanduser()


def count(kind: str, n: int = 1) -> None:
    with _lock:
        STATS[kind] += n


@cache
def agent() -> str:
    from importlib.metadata import version  # slow to import: only commands that fetch pay it

    return f"mpc-cli/{version('mpc-cli')}"


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": agent()})
    return _client


def fetch(method: str, url: str, *, params=None, body=None, tries: int = 4) -> httpx.Response:
    for k in range(tries):
        last = k == tries - 1
        try:
            r = client().request(method, url, params=params, json=body)
        except httpx.TransportError:
            if last:
                raise
            time.sleep(2**k)
            continue
        count("requests")
        if r.status_code == 429 or r.status_code >= 500:
            if last:
                r.raise_for_status()
            time.sleep(min(float(r.headers.get("retry-after", 2**k)), 60))
            continue
        r.raise_for_status()
        return r
    raise AssertionError("unreachable")


def cached_json(method: str, url: str, *, params=None, body=None, ttl: float | None = None):
    """The JSON answer, from the cache while younger than `ttl` seconds (None: always)."""
    key = hashlib.sha256(json.dumps([method, url, params, body], sort_keys=True).encode())
    path = root() / "http" / key.hexdigest()[:2] / f"{key.hexdigest()}.json"
    if path.exists() and (ttl is None or time.time() - path.stat().st_mtime < ttl):
        count("hits")
        return json.loads(path.read_text())
    data = fetch(method, url, params=params, body=body).json()
    write_atomic(path, json.dumps(data).encode())
    return data


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
