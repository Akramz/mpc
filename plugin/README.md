# mpc

A Claude Code plugin that reads satellite imagery and maps from the Planetary Computer at a place,
at native resolution: values over time, images, band combinations and indices, land cover classes
and elevation. It is an independent project, not a Microsoft product.

```console
/plugin install mpc --marketplace Akramz/mpc
```

The plugin is one skill. It runs the `mpc` command line tool, and when `mpc` is not installed it
runs the pinned release from PyPI with `uvx`, so it needs only [uv](https://docs.astral.sh/uv/).

## Data and network

The skill sends data to these services, and no others:

- `planetarycomputer.microsoft.com`: the place (longitude, latitude), dates and band names you ask
  about, for STAC scene search, pixel values from the data API, and read-only SAS tokens.
- `*.blob.core.windows.net` (Azure Blob Storage): read-only requests for the public satellite image
  files the Planetary Computer's scenes point to.
- `pypi.org` and `files.pythonhosted.org`: only when `mpc` is not installed, `uvx` downloads the
  pinned `mpc-cli` release.

It sends no personal data, credentials or telemetry, and it runs no service of its own. What it
reads is cached on your machine in `~/.cache/mpc` until you delete it; it writes other files only
where you ask it to.

Source, documentation and issues: https://github.com/Akramz/mpc. MIT license.
