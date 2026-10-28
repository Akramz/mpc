# mpc

A Claude Code plugin that reads satellite imagery and maps from the Planetary Computer at a place,
at native resolution: values over time, images, band combinations and indices, land cover classes
and elevation. It is an independent project, not a Microsoft product.

```console
/plugin install mpc --marketplace Akramz/mpc
```

The plugin is one skill. It runs the `mpc` command line tool, and when `mpc` is not installed it
runs the pinned release from PyPI with `uvx`, so it needs only [uv](https://docs.astral.sh/uv/).

mpc talks only to the Planetary Computer: `planetarycomputer.microsoft.com` (STAC search, the data
API and SAS tokens) and the Azure Blob Storage its items point to (`*.blob.core.windows.net`). It
sends no telemetry. It keeps what it reads in `~/.cache/mpc` on your machine, and writes files only
where you ask it to.

Source, documentation and issues: https://github.com/Akramz/mpc. MIT license.
