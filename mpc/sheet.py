"""One PNG sheet for each read: dates side by side, native pixels shown with nearest zoom, the
pixel that holds the point outlined in red. Some agent CLIs pass one image per request."""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEADER_PX = 28
OUT_PX = 448
GRID = (255, 230, 0)  # quadrant lines, digits and ticks of a navigation view
COLORS = [  # class colors: (name for the caption, RGB)
    ("blue", (31, 119, 180)), ("orange", (255, 127, 14)), ("green", (44, 160, 44)),
    ("red", (214, 39, 40)), ("purple", (148, 103, 189)), ("brown", (140, 86, 75)),
    ("pink", (227, 119, 194)), ("gray", (127, 127, 127)), ("olive", (188, 189, 34)),
    ("cyan", (23, 190, 207)), ("light blue", (174, 199, 232)), ("light orange", (255, 187, 120)),
    ("light green", (152, 223, 138)), ("light red", (255, 152, 150)), ("lavender", (197, 176, 213)),
    ("tan", (196, 156, 148)), ("light pink", (247, 182, 210)), ("light gray", (199, 199, 199)),
    ("khaki", (219, 219, 141)), ("light cyan", (158, 218, 229)),
]  # fmt: skip


def color(i: int) -> tuple[str, tuple[int, int, int]]:
    return COLORS[i % len(COLORS)]


def stretch(stack: np.ndarray) -> np.ndarray:
    """(T, C, H, W) float with NaN to (T, H, W, 3) uint8: one 2..98 percentile stretch over every
    date and channel, so dates compare and a false-color composite keeps its band ratios. NaN
    renders black; grayscale is replicated."""
    ok = np.isfinite(stack)
    out = np.zeros(stack.shape, dtype=np.uint8)
    if ok.any():
        a, b = np.percentile(stack[ok], [2, 98])
        b = b if b > a else a + 1e-6
        y = np.clip((np.nan_to_num(stack, nan=a) - a) / (b - a), 0, 1)
        out = np.where(ok, y * 255, 0).astype(np.uint8)
    rgb = out.transpose(0, 2, 3, 1)
    return np.repeat(rgb, 3, axis=3) if rgb.shape[3] == 1 else rgb


def classes(stack: np.ndarray, values: list[int]) -> np.ndarray:
    """(T, 1, H, W) class values to (T, H, W, 3) colors, one per value in `values` order."""
    out = np.zeros((*stack.shape[:1], *stack.shape[2:], 3), dtype=np.uint8)
    for i, v in enumerate(values):
        out[stack[:, 0] == v] = color(i)[1]
    return out


def chip(rgb: np.ndarray) -> Image.Image:
    """An (H, W, 3) image resized to OUT_PX (nearest up, box down) with a 2 px red frame
    around the center pixel."""
    out_px = OUT_PX
    h, w = rgb.shape[:2]
    img = Image.fromarray(np.ascontiguousarray(rgb))
    if (h, w) != (out_px, out_px):
        method = Image.Resampling.NEAREST if out_px >= max(h, w) else Image.Resampling.BOX
        img = img.resize((out_px, out_px), method)
    side = max(int(round(out_px / w)), 1)
    a = (out_px - side) // 2
    ImageDraw.Draw(img).rectangle(
        [a - 2, a - 2, a + side + 1, a + side + 1], outline=(255, 0, 0), width=2
    )
    return img


def marked(rgb: np.ndarray, place: tuple[float, float] | None) -> bytes:
    """A navigation view as PNG: the (H, W, 3) image at OUT_PX at least (nearest), quadrant
    lines and digits, a tick every 10 % on the top and left edges, and a red box at `place`
    (x, y in pixels of `rgb`) when it is inside."""
    img = Image.fromarray(np.ascontiguousarray(rgb))
    if img.width < OUT_PX:
        img = img.resize((OUT_PX, OUT_PX), Image.Resampling.NEAREST)
    d, n = ImageDraw.Draw(img), img.width
    m = n // 2
    big = dict(fill=GRID, font=ImageFont.load_default(size=26), stroke_width=2, stroke_fill=0)
    small = dict(fill=GRID, font=ImageFont.load_default(size=13), stroke_width=1, stroke_fill=0)
    d.line([(m, 0), (m, n)], fill=GRID, width=2)
    d.line([(0, m), (n, m)], fill=GRID, width=2)
    for k, (x, y) in enumerate([(22, 18), (m + 22, 18), (22, m + 18), (m + 22, m + 18)], 1):
        d.text((x, y), str(k), **big)
    for p in range(10, 100, 10):
        t = n * p / 100
        d.line([(t, 0), (t, 6)], fill=GRID, width=2)
        d.line([(0, t), (6, t)], fill=GRID, width=2)
        d.text((t + 2, 1), str(p), **small)
        d.text((1, t + 2), str(p), **small)
    if place is not None:
        x, y = (c * n / rgb.shape[1] for c in place)
        d.rectangle([x - 8, y - 8, x + 8, y + 8], outline=(255, 0, 0), width=2)
    return png(img)


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def sheet(tiles: list[tuple[str, np.ndarray]]) -> bytes:
    """Every (label, rgb) tile on a near-square grid, each under a strip with its label."""
    n = len(tiles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    out_px, gap, head = OUT_PX, 8, HEADER_PX
    tile_h = out_px + head
    size = (cols * out_px + (cols - 1) * gap, rows * tile_h + (rows - 1) * gap)
    out = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=20)
    for i, (lab, rgb) in enumerate(tiles):
        x = (i % cols) * (out_px + gap)
        y = (i // cols) * (tile_h + gap)
        draw.text((x + 4, y + 3), lab, fill=(0, 0, 0), font=font)
        out.paste(chip(rgb), (x, y + head))
    return png(out)
