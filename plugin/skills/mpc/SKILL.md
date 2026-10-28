---
name: mpc
description: Read satellite imagery and maps from Microsoft Planetary Computer at a place, at native resolution, with the `mpc` CLI (values over time, images, band combinations and indices, land cover classes, elevation). Use when the user asks what is at a location, how a place changed, which data covers it, or wants imagery pixels for later work.
---

# mpc

`mpc` reads the Microsoft Planetary Computer (MPC). If `mpc` is not found, run
`uvx --from mpc-cli==0.1.0 mpc` in its place, with the same arguments; it needs no install. If
`uvx` is not found either, tell the user to install uv (https://docs.astral.sh/uv/) and stop.

## Find data

- `mpc catalog <words>`: collections that match, with years and resolution. No network call.
- `mpc info <collection>`: its asset names, resolution, units, class values and MPC's color
  presets.
- `mpc search <collection> --location LON,LAT --period START END [--json]`: each scene at the
  place, with its STAC properties (cloud cover, where the product has it). Properties that are
  the same for every scene print once; scene cloud cover is for the whole scene, not the point.
  Use it when you don't know the dates; known dates go straight to `mpc read`.

`--expr` takes asset names and expressions over them, as many as you want in one call:
`--expr B04 SCL "(B08-B04)/(B08+B04)"`. Comparisons (`== != < <= > >=`) give 1 or 0, and no
value where data is missing, so `"SCL==6"` maps water and `"(B08-B04)/(B08+B04) > 0.6"` dense
vegetation. Use asset names exactly as `mpc info` lists them (`B08`, not `nir`). A multi-band asset is `<asset>_b<n>`, e.g. NAIP's near-infrared is `image_b4`.

## Look

Use `navigate` when you must find a place by eye across a whole scene (a lake, a field edge, a
burn scar). If you know the place, read it directly.

```
mpc navigate <collection> --location LON,LAT --date YYYY-MM-DD --expr A [B C]
mpc navigate --view ID --path 2/4/1 [--expr A [B C]]
```

- Open a scene with its date. The first view is the whole scene at its coarsest overview.
  Give one expression (grayscale) or three (RGB); `mpc info` lists MPC's color presets, such
  as true color. Each new view costs about 2 seconds; a view seen before is free.
- Each view prints its id first. `--view ID --path P` shows path P of that view's scene.
- A path counts from the whole scene: quadrant digits 1 top-left, 2 top-right, 3 bottom-left,
  4 bottom-right, joined by "/". Add a digit to zoom in, drop one to zoom out; "" is the whole
  scene. `--expr` changes what a view shows.
- Open the PNG and read the caption. Ticks mark every 10 % of the view from the left and top.
- Read a spot you found with `mpc read <collection> --view ID --at X,Y`, X and Y percent from
  the left and top of that view. `--view ID --at X,Y` replaces `--location`; the rest of `read`
  is unchanged, so give `--expr` and `--dates` as usual.
- Views are for finding places. Values come from `mpc read`.

## Read

```
mpc read <collection> (--location LON,LAT | --view ID --at X,Y) --expr EXPR [EXPR ...]
         (--dates YYYY-MM-DD ... | --period START END) [--buffer METERS] [--save FILE.tif] [--json]
```

- Without `--buffer`: a table, one row for each image, of the native pixel at the location.
- With `--buffer`: a native-resolution image around the location, dates side by side, the
  location's pixel outlined in red. `mpc` prints a caption and a PNG path: open the PNG with your
  image tool and read the caption. One or three expressions; at most 12 dates.
- `--dates`: the scene on each date, exactly as `mpc search` lists it. A date with no scene is
  an error. `--period`: every scene from START to END, both included, the same scenes `mpc
  search` lists. A scene can have no data at the place; its row shows "-".
- `--save FILE.tif` (with `--buffer`) writes the native window as one GeoTIFF stack, bands named
  `{date}_{expression}`, for the user's later work. It accepts any number of dates and
  expressions. To analyze it, run Python with rasterio: `uv run --with rasterio python script.py`.
- Sentinel-2 bands come as surface reflectance (0 to 1); other collections apply the scale and
  offset their catalog states. Class maps print the class name beside each value.

Pick dates from `mpc search` when you don't know them. Ask for every date and expression you want
to compare in one call, and prefer one `--period` call to many calls. Scene cloud cover says little
about one pixel: check `SCL` (Sentinel-2) or the image before you trust a value. Name the
dates (and with `--json`, the item ids) your answer rests on.

## The store

Every pixel `mpc` reads stays in `~/.cache/mpc`, so a repeat or a smaller window inside an
earlier one costs nothing. Never prune on your own. If the user asks about space, run
`mpc store du` (largest first, by collection, region and period), show it, and run
`mpc store prune --collection ... --bbox W,S,E,N --period START END` only for what the user names.
`mpc store ls --bbox W,S,E,N` lists what is stored there. When mpc says the store holds more than
10 GB, tell the user and show `mpc store du`.
