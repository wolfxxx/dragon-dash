"""Rebuild clean Pip idle/run sheets for Dragon Dash."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

BASE = Path(__file__).resolve().parent
CELL_W = 420
CELL_H = 300
PAD = 18


def content_mask(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3]
    return (rgb.sum(axis=2) > 36) & (alpha > 24)


def scrub_black(im: Image.Image) -> Image.Image:
    arr = np.array(im)
    near_black = arr[:, :, :3].max(axis=2) < 18
    arr[:, :, 3] = np.where(near_black, 0, arr[:, :, 3])
    return Image.fromarray(arr, "RGBA")


def largest_component(im: Image.Image, min_pixels: int = 80) -> Image.Image:
    arr = np.array(im)
    mask = content_mask(arr)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for y in range(h):
        for x in np.where(mask[y] & ~visited[y])[0]:
            stack = [(y, int(x))]
            visited[y, x] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    keep = np.zeros_like(mask, dtype=bool)
    if len(best) >= min_pixels:
        for cy, cx in best:
            keep[cy, cx] = True
    arr[:, :, 3] = np.where(keep, arr[:, :, 3], 0)
    return Image.fromarray(arr, "RGBA")


def tight_crop(im: Image.Image, pad: int = 4) -> Image.Image:
    arr = np.array(im)
    mask = content_mask(arr)
    if not mask.any():
        return im
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    return im.crop(
        (
            max(0, int(xs[0]) - pad),
            max(0, int(ys[0]) - pad),
            min(im.width - 1, int(xs[-1]) + pad) + 1,
            min(im.height - 1, int(ys[-1]) + pad) + 1,
        )
    )


def extract_left_idle(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    crop = im.crop((0, 240, 575, 660))
    return tight_crop(largest_component(scrub_black(crop)), pad=6)


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in np.where(mask[y] & ~visited[y])[0]:
            stack = [(y, int(x))]
            visited[y, x] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(comp)
    return comps


def make_blink(frame: Image.Image) -> Image.Image:
    """Close the orange eye near the snout; ignore horn gold."""
    out = frame.copy()
    arr = np.array(out)
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    head = (
        (np.arange(h)[:, None] > int(h * 0.18))
        & (np.arange(h)[:, None] < int(h * 0.36))
        & (np.arange(w)[None, :] > int(w * 0.72))
    )
    orange = (
        head
        & (r > 200)
        & (g > 80)
        & (g < 210)
        & (b < 90)
        & ((r - b) > 120)
        & (alpha > 80)
    )
    pupil = head & (r < 90) & (g < 80) & (b < 70) & (alpha > 80)
    comps = [c for c in components(orange) if 12 <= len(c) <= 1500]
    if not comps:
        return out

    # Eye sits on the snout side (~88% width, ~29% height) and near a pupil.
    ax, ay = w * 0.88, h * 0.29

    def score(comp: list[tuple[int, int]]) -> float:
        ys = np.array([p[0] for p in comp])
        xs = np.array([p[1] for p in comp])
        cy, cx = float(ys.mean()), float(xs.mean())
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        local_pupil = pupil[max(0, y0 - 4) : y1 + 5, max(0, x0 - 4) : x1 + 5].sum()
        dist = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
        return dist - local_pupil * 0.8

    best = min(comps, key=score)
    ys = np.array([p[0] for p in best])
    xs = np.array([p[1] for p in best])
    cy, cx = int(ys.mean()), int(xs.mean())
    # Cover the full socket (iris + lids), not just the brightest orange core.
    x0, x1 = cx - 22, cx + 24
    y0, y1 = cy - 14, cy + 16

    # Sample green brow / cheek scales around the socket.
    sample = arr[max(0, cy - 22) : cy - 8, max(0, cx - 20) : cx + 12]
    opaque = sample[:, :, 3] > 80 if sample.size else None
    greenish = None
    if opaque is not None and np.any(opaque):
        cols = sample[:, :, :3][opaque]
        greener = cols[:, 1] > cols[:, 0] + 10
        pick = cols[greener] if greener.any() else cols
        greenish = tuple(int(c) for c in pick.mean(axis=0))
    color = greenish or (34, 118, 56)
    dark = tuple(max(0, int(c * 0.55)) for c in color)

    # Soft feathered lid so it reads as a blink instead of a sticker.
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([x0, y0, x1, y1], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.6))

    lid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(lid)
    draw.ellipse([x0, y0, x1, y1], fill=color + (255,))
    draw.ellipse([x0 + 4, cy - 4, x1 - 4, cy + 7], fill=dark + (255,))
    draw.line([(x0 + 6, cy + 1), (x1 - 6, cy + 1)], fill=(8, 40, 16, 255), width=2)
    la = np.array(lid)
    ma = np.array(mask)
    la[:, :, 3] = (la[:, :, 3].astype(np.float32) * ma / 255.0).astype(np.uint8)
    out.alpha_composite(Image.fromarray(la, "RGBA"))
    print(f"blink eye at ({cx},{cy}) box=({x0},{y0})-({x1},{y1}) color={color}")
    return scrub_black(out)


def breath_variant(frame: Image.Image, scale_y: float, brightness: float) -> Image.Image:
    """Visible chest breath via vertical scale around feet."""
    arr = np.array(frame)
    mask = content_mask(arr)
    ys = np.where(mask.any(axis=1))[0]
    feet = int(ys[-1])
    new_h = max(1, int(round(frame.height * scale_y)))
    scaled = frame.resize((frame.width, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (frame.width, frame.height + 20), (0, 0, 0, 0))
    # Keep feet roughly planted.
    dest_y = 10 + feet - int(round(feet * scale_y))
    canvas.alpha_composite(scaled, (0, dest_y))
    return ImageEnhance.Brightness(tight_crop(canvas, pad=2)).enhance(brightness)


def density_peaks(mask: np.ndarray, n: int) -> list[int]:
    dens = mask.sum(axis=0).astype(np.float64)
    smooth = np.convolve(dens, np.ones(61) / 61, mode="same")
    smooth[:25] = 0
    smooth[-25:] = 0
    peaks: list[int] = []
    work = smooth.copy()
    min_sep = mask.shape[1] // (n + 1)
    for _ in range(n):
        x = int(np.argmax(work))
        peaks.append(x)
        work[max(0, x - min_sep) : min(len(work), x + min_sep)] = 0
    return sorted(peaks)


def split_run_frames(path: Path, n: int = 4) -> list[Image.Image]:
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    mask = content_mask(arr)
    peaks = density_peaks(mask, n)
    w = im.width
    owners = np.array([int(np.argmin([abs(x - p) for p in peaks])) for x in range(w)])
    frames: list[Image.Image] = []
    for i in range(n):
        cols = np.where(owners == i)[0]
        x0, x1 = int(cols[0]), int(cols[-1])
        band = mask[:, x0 : x1 + 1]
        xs = np.where(band.any(axis=0))[0]
        ys = np.where(band.any(axis=1))[0]
        crop = scrub_black(
            im.crop(
                (
                    max(0, x0 + int(xs[0]) - 2),
                    max(0, int(ys[0]) - 2),
                    min(w - 1, x0 + int(xs[-1]) + 2) + 1,
                    min(im.height - 1, int(ys[-1]) + 2) + 1,
                )
            )
        )
        frames.append(tight_crop(largest_component(crop), pad=2))
    return frames


def place(frame: Image.Image, scale: float) -> Image.Image:
    nw = max(1, int(round(frame.width * scale)))
    nh = max(1, int(round(frame.height * scale)))
    scaled = largest_component(scrub_black(frame.resize((nw, nh), Image.Resampling.LANCZOS)))
    arr = np.array(scaled)
    mask = content_mask(arr)
    ys = np.where(mask.any(axis=1))[0]
    feet = int(ys[-1])
    # Anchor on the feet band so breath scaling doesn't slide the dragon sideways.
    foot_band = mask[max(0, feet - 10) : feet + 1]
    foot_xs = np.where(foot_band.any(axis=0))[0]
    if foot_xs.size:
        cx = int((foot_xs[0] + foot_xs[-1]) / 2)
    else:
        xs = np.where(mask.any(axis=0))[0]
        cx = int((xs[0] + xs[-1]) / 2)
    cell = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
    dest_x = int(np.clip(CELL_W // 2 - cx, 0, max(0, CELL_W - nw)))
    dest_y = int(np.clip(CELL_H - PAD - 1 - feet, 0, max(0, CELL_H - nh)))
    cell.alpha_composite(scaled, (dest_x, dest_y))
    return scrub_black(cell)


def build_sheet(frames: list[Image.Image], out: Path) -> dict:
    heights = []
    for f in frames:
        m = content_mask(np.array(f))
        ys = np.where(m.any(axis=1))[0]
        heights.append(ys[-1] - ys[0] + 1)
    target = float(np.median(heights))
    max_h = CELL_H - PAD * 2
    max_w = CELL_W - PAD * 2
    scale = min(max_h / target, max_w / max(f.width for f in frames))
    cells = [place(f, scale) for f in frames]
    sheet = Image.new("RGBA", (CELL_W * len(cells), CELL_H), (0, 0, 0, 0))
    for i, c in enumerate(cells):
        sheet.paste(c, (i * CELL_W, 0))
    sheet.save(out)
    return {"path": out.name, "frames": len(cells), "cell": (CELL_W, CELL_H), "scale": round(scale, 4)}


def preview(sheet_path: Path, preview_path: Path) -> None:
    sheet = Image.open(sheet_path).convert("RGBA")
    prev = Image.new("RGBA", sheet.size, (40, 44, 70, 255))
    for y in range(0, sheet.height, 16):
        for x in range(0, sheet.width, 16):
            if ((x // 16) + (y // 16)) % 2 == 0:
                for yy in range(y, min(y + 16, sheet.height)):
                    for xx in range(x, min(x + 16, sheet.width)):
                        prev.putpixel((xx, yy), (55, 60, 90, 255))
    prev.alpha_composite(sheet)
    d = ImageDraw.Draw(prev)
    for i in range(1, sheet.width // CELL_W):
        d.line([(i * CELL_W, 0), (i * CELL_W, sheet.height)], fill=(255, 220, 80, 160), width=1)
    prev.convert("RGB").save(preview_path, quality=92)


def main() -> None:
    base = extract_left_idle(BASE / "pip-idle-final.png")
    blink = make_blink(base)
    base.save(BASE / "_pip-idle-base.png")
    blink.save(BASE / "_pip-idle-blink.png")

    idle_frames = [
        base,
        breath_variant(base, 1.045, 1.03),
        blink,
        breath_variant(base, 0.97, 0.98),
    ]

    run_frames = split_run_frames(BASE / "pip-run.png", 4)
    idle_info = build_sheet(idle_frames, BASE / "pip-idle-sheet.png")
    run_info = build_sheet(run_frames, BASE / "pip-run-sheet.png")
    preview(BASE / "pip-idle-sheet.png", BASE / "_preview-idle-sheet.jpg")
    preview(BASE / "pip-run-sheet.png", BASE / "_preview-run-sheet.jpg")
    print("base", base.size)
    print("idle", idle_info)
    print("run", run_info)


if __name__ == "__main__":
    main()
