"""Read the Microsoft Planetary Computer at a place, at native resolution."""


def read(collection: str, lon: float, lat: float, **kw):
    """`mpc.read("sentinel-2-l2a", 34.4, 0.49, exprs=("B04",), period=("2023-01-01", "2023-12-31"),
    buffer_m=500)`: a Result with `values` (T, E) or `planes` (T, E, H, W), `crs`, `transform`."""
    from mpc.reader import Read, run

    return run(Read(collection, lon, lat, **kw))
