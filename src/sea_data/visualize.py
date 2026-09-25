"""Visual verification: draw ground-truth objects and horizons onto images.

Renders each ``meta/*.json`` frame's object boxes and horizon line on top of
its photo, so a human can eyeball whether the annotations actually line up
with what was rendered. Uses Pillow because the rendered photos come from
Blender's own PNG encoder, which this project's dependency-free PNG writer
(:func:`sea_data.annotations.write_png`, used only for the masks it writes
itself) cannot decode.
"""

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from sea_data.annotations import iter_frames
from sea_data.config import CLASS_NAMES

OVERLAY_DIR = "debug"

PANEL_WIDTH = 520
PANEL_PADDING = 16
PANEL_FONT_SIZE = 22
LINE_HEIGHT = 30
ROW_GAP = 6
PANEL_BG = (24, 24, 24)
PANEL_FG = (230, 230, 230)

_PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
)
INVALID_COLOR = (140, 140, 140)
AMODAL_COLOR = (255, 255, 255)
HORIZON_COLOR = (255, 255, 0)


def category_color(category: str) -> tuple[int, int, int]:
    """Return a stable colour for an object category.

    Args:
        category: Object class name.

    Returns:
        An RGB colour.
    """
    return _PALETTE[CLASS_NAMES.index(category) % len(_PALETTE)]


def draw_object(draw: ImageDraw.ImageDraw, obj: Mapping[str, Any]) -> None:
    """Draw one object's boxes and label onto an overlay.

    The amodal (unoccluded) box is drawn in white, the visible box in a
    category colour when the object is labelled, or grey when it was
    filtered out as not visible enough to train on.

    Args:
        draw: Drawing context of the overlay image.
        obj: Object record from a frame's metadata.
    """
    amodal = obj.get("amodal_bbox")
    if amodal:
        draw.rectangle(amodal, outline=AMODAL_COLOR, width=1)
    box = obj.get("bbox")
    if not box:
        return
    color = category_color(obj["category"]) if obj["valid"] else INVALID_COLOR
    draw.rectangle(box, outline=color, width=2)
    label = f"{obj['category']} {obj['distance_m']:.0f}m"
    draw.text((box[0], max(box[1] - 12, 0)), label, fill=color)


def draw_horizon(draw: ImageDraw.ImageDraw, horizon: Mapping[str, Any]) -> None:
    """Draw the horizon line onto an overlay, if visible.

    Args:
        draw: Drawing context of the overlay image.
        horizon: Horizon record from a frame's metadata.
    """
    endpoints = horizon.get("endpoints")
    if not horizon["visible"] or endpoints is None:
        return
    (x0, y0), (x1, y1) = endpoints
    draw.line([(x0, y0), (x1, y1)], fill=HORIZON_COLOR, width=2)


def render_overlay(output_dir: Path, frame: Mapping[str, Any]) -> Image.Image:
    """Draw a frame's ground-truth objects and horizon onto its rendered photo.

    Args:
        output_dir: Dataset root directory.
        frame: Per-image metadata.

    Returns:
        A copy of the rendered image with annotations drawn on top.
    """
    image = Image.open(output_dir / frame["image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw_horizon(draw, frame["horizon"])
    for obj in frame["objects"]:
        draw_object(draw, obj)
    return image


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def frame_metadata_lines(frame: Mapping[str, Any]) -> list[str]:
    """Format a frame's horizon, atmosphere and object fields as text lines.

    Args:
        frame: Per-image metadata.

    Returns:
        Lines to print top to bottom, grouped by section.
    """
    horizon = frame["horizon"]
    atmosphere = frame.get("scenario", {}).get("atmosphere", {})
    lines = [
        Path(frame["image"]).name,
        f"{frame['width']}x{frame['height']}",
        "",
        "Horizon",
        f"  visible: {horizon['visible']}",
        f"  transmittance: {_fmt(horizon.get('transmittance'), '.3g')}",
        f"  mask_error_px: {_fmt(horizon.get('mask_error_px'))}",
        f"  mask_agreement: {_fmt(horizon.get('mask_agreement'))}",
        "",
        "Atmosphere",
        f"  weather: {atmosphere.get('weather', '-')}",
        f"  visibility_m: {_fmt(atmosphere.get('visibility_m'), '.0f')}",
        f"  cloud_cover: {_fmt(atmosphere.get('cloud_cover'), '.2f')}",
        f"  aerosol_density: {_fmt(atmosphere.get('aerosol_density'), '.2f')}",
        "",
        f"Objects ({len(frame['objects'])})",
    ]
    for obj in frame["objects"]:
        mark = "OK" if obj["valid"] else "--"
        lines.append(
            f"  [{mark}] {obj['category']} {obj['distance_m']:.0f}m "
            f"px={obj['visible_pixels']}"
        )
    return lines


def _draw_panel(canvas: Image.Image, x: int, lines: Sequence[str]) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x, 0, canvas.width, canvas.height], fill=PANEL_BG)
    font = ImageFont.load_default(size=PANEL_FONT_SIZE)
    y = PANEL_PADDING
    for line in lines:
        draw.text((x + PANEL_PADDING, y), line, fill=PANEL_FG, font=font)
        y += LINE_HEIGHT


def render_comparison(output_dir: Path, frame: Mapping[str, Any]) -> Image.Image:
    """Stack a frame's original photo above its annotated overlay.

    A side panel lists the horizon, atmosphere and per-object ground truth,
    so the raw photo, the drawn boxes/horizon and the numbers behind them can
    all be checked together.

    Args:
        output_dir: Dataset root directory.
        frame: Per-image metadata.

    Returns:
        The combined comparison image.
    """
    original = Image.open(output_dir / frame["image"]).convert("RGB")
    annotated = render_overlay(output_dir, frame)
    width, height = original.size
    canvas = Image.new("RGB", (width + PANEL_WIDTH, 2 * height + ROW_GAP), PANEL_BG)
    canvas.paste(original, (0, 0))
    canvas.paste(annotated, (0, height + ROW_GAP))
    _draw_panel(canvas, width, frame_metadata_lines(frame))
    return canvas


def visualize_dataset(
    output_dir: Path, overlay_dir: Path | None = None, limit: int | None = None
) -> list[Path]:
    """Write original/annotated comparison images for a rendered dataset.

    Each output stacks the original photo above the annotated overlay, with
    a side panel of horizon, atmosphere and object ground truth.

    Args:
        output_dir: Dataset root directory containing ``meta/``.
        overlay_dir: Folder to write images into; defaults to
            ``output_dir / "debug"``.
        limit: Maximum number of images to write; ``None`` writes all.

    Returns:
        Paths of the images written, in frame order.
    """
    target = overlay_dir or output_dir / OVERLAY_DIR
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for frame in itertools.islice(iter_frames(output_dir), limit):
        comparison = render_comparison(output_dir, frame)
        path = target / Path(frame["image"]).name
        comparison.save(path)
        written.append(path)
    return written
