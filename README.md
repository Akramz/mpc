# mpc

`mpc` reads the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com) at a
place: the value of a band or an index over time, or a native-resolution image around the place.
It is built for coding agents (Claude Code, Copilot CLI, Codex) and works the same for people.
Everything it reads stays in a local store you can query again or use for later work.

mpc is an independent project. It is not a Microsoft product, and Microsoft does not endorse
or support it.

## Install

```console
uv tool install mpc-cli
mpc install-skill        # links the skill into ~/.claude, ~/.copilot and ~/.codex, where present
```

Upgrade with `uv tool upgrade mpc-cli`. The skill is a link, so it follows the upgrade.

## Use

```console
$ mpc catalog land cover
$ mpc info sentinel-2-l2a
$ mpc search sentinel-2-l2a --location 34.39767,0.48864 --period 2023-05-01 2023-05-31
$ mpc navigate sentinel-2-l2a --location 34.39767,0.48864 --date 2023-05-16 --expr B04 B03 B02
$ mpc navigate --view 574087be5312 --path 2/3
$ mpc read sentinel-2-l2a --view edc73ccf520c --at 62,35 --expr "(B08-B04)/(B08+B04)" --dates 2023-05-16
$ mpc read sentinel-2-l2a --location 34.39767,0.48864 --expr "(B08-B04)/(B08+B04)" \
    --dates 2023-02-15 2023-05-16 2023-08-14
date        (B08-B04)/(B08+B04)
2023-02-15                0.301
2023-05-16                0.764
2023-08-14                0.338
$ mpc read sentinel-2-l2a --location 34.39767,0.48864 --expr B04 B03 B02 --dates 2023-05-16 --buffer 400
$ mpc read sentinel-2-l2a --location 34.39767,0.48864 --expr B02 B03 B04 B08 \
    --period 2023-01-01 2023-12-31 --buffer 2000 --save bungoma_2023.tif
$ mpc store du
```

In Python: `import mpc; r = mpc.read("sentinel-2-l2a", 34.4, 0.49, exprs=("B08",), dates=("2023-05-16",))`.

## How it reads

- **The catalog** ships in the package: one short record for each of the 139 collections (78
  have COG assets mpc reads). `mpc catalog --refresh` rebuilds it from the live catalog; nothing
  refreshes on its own.
- **Scenes.** `mpc search` lists each scene at the place in a period, with its STAC properties:
  the ones every scene shares once, the rest as columns (`--json` gives them all). Cloud cover,
  where a product has it, is for the whole scene, not the place.
- **Dates.** `read` and `navigate` need them: a date gets the scene on that day (an acquisition
  that UTC day, or an interval that holds it, like an annual map) or fails; mpc never swaps in
  another date. A period gets every scene in it (both ends included). Both see exactly the
  scenes `mpc search` lists. Overlapping tiles give two items for one acquisition; mpc keeps the
  first by id.
- **Inputs.** `read` and a new view need `--expr`: asset names or expressions over them, as
  many as you want (an image shows one or three). There are no defaults. Expressions take
  `+ - * / **`, `abs sqrt log exp`, and comparisons that give 1 or 0 (`"SCL==6"`).
- **Views** (`mpc navigate`) come from the COG overviews of one scene: zoom 0 is the whole
  scene at its coarsest overview, and each quadrant digit of a path (1 top-left, 2 top-right,
  3 bottom-left, 4 bottom-right) halves the view and reads the next finer overview, down to
  native pixels. Each view prints an id; `mpc navigate --view ID --path P` moves within its
  scene, and `mpc read --view ID --at X,Y` reads the spot X and Y percent from its left and top.
  There is no current view, so agents that run at once never share one. A view's caption
  lists its scene's overview levels.
- **Values at a place** come from MPC's data API: the native pixel, about 100 bytes, no token.
- **Windows** (`--buffer`) come straight from the COGs, one 512-pixel internal block at a time,
  with one SAS token for each collection, renewed before it expires and after a 403. No
  reprojection, no resampling. An expression over 10 m and 20 m bands puts the 20 m pixels 2 x 2
  on the 10 m grid, and the caption says so.
- **Units.** Sentinel-2 bands become reflectance (DN / 10000, minus 0.1 from processing baseline
  04.00) with 0 as no data; other collections use the scale, offset and nodata their catalog
  states.

## The store

`~/.cache/mpc` (or `$MPC_HOME`) holds `store.sqlite`, an index of items, asset grids, blocks and
point values, and `store/<collection>/<item>/<asset>/r<row>_c<col>.tif`, one GeoTIFF for each
block. A window inside an earlier one reads only stored blocks. Nothing expires; `mpc store du`
ranks what is stored by collection, region and period, and `mpc store prune` removes what you
name. Past 10 GB, `read` and `navigate` say so. STAC searches are cached for an hour.

## Network and files

mpc talks only to the Planetary Computer: `planetarycomputer.microsoft.com` (STAC search, the
data API and SAS tokens) and the Azure Blob Storage its items point to
(`*.blob.core.windows.net`). It sends no telemetry. It writes to `~/.cache/mpc` (or
`$MPC_HOME`), to files you name with `--save`, and, with `mpc install-skill`, to a link in each
of `~/.claude`, `~/.copilot` and `~/.codex` that exists.

## Development

```console
make verify    # lint and offline tests (about a second)
make live      # compare mpc with plain rasterio reads at four places, on the live MPC
make catalog   # rebuild the snapshot in mpc/catalog.json
make bench     # time a headless Claude agent on bench/tasks.json; compare runs across changes
```
