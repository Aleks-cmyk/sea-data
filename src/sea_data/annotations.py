"""Dataset export: mask images and COCO / YOLO / horizon annotation files.

The Blender worker writes one ``meta/<stem>.json`` file per rendered image.
:func:`build_index` gathers those files into the dataset-level formats that
training frameworks consume. Nothing in this module depends on Blender.
"""

import csv
import json
import struct
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sea_data.config import (
    CLASS_NAMES,
    SEMANTIC_OBJECT_OFFSET,
    SEMANTIC_SEA,
    SEMANTIC_SKY,
)

type BBox = tuple[float, float, float, float]
"""Axis-aligned box ``(x_min, y_min, x_max, y_max)`` in pixel edges."""

META_DIR = "meta"
IMAGE_DIR = "images"
LABEL_DIR = "labels"
MASK_DIR = "masks"
COCO_FILE = "annotations_coco.json"
HORIZON_FILE = "horizon.csv"
YOLO_DATASET_FILE = "dataset.yaml"

HORIZON_COLUMNS = (
    "image",
    "visible",
    "x0",
    "y0",
    "x1",
    "y1",
    "y_left",
    "y_right",
    "angle_deg",
    "offset_px",
    "dip_deg",
    "camera_height_m",
    "camera_pitch_deg",
    "camera_roll_deg",
    "transmittance",
)


def write_png(
    path: Path, width: int, height: int, data: bytes, channels: int = 1
) -> None:
    """Write an 8-bit PNG image without third-party dependencies.

    Args:
        path: Output file.
        width: Image width in pixels.
        height: Image height in pixels.
        data: Row-major pixel bytes, top row first, ``channels`` per pixel.
        channels: 1 (grey), 3 (RGB) or 4 (RGBA).

    Raises:
        ValueError: If ``data`` has the wrong size or ``channels`` is invalid.
    """
    color_types = {1: 0, 3: 2, 4: 6}
    try:
        color_type = color_types[channels]
    except KeyError:
        raise ValueError(f"unsupported channel count: {channels}") from None
    stride = width * channels
    if len(data) != stride * height:
        raise ValueError(f"expected {stride * height} bytes, got {len(data)}")
    raw = b"".join(
        b"\x00" + data[row * stride : (row + 1) * stride] for row in range(height)
    )

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def semantic_value(category: str) -> int:
    """Return the semantic-mask value used for an object class.

    Args:
        category: Object class name.

    Returns:
        The 8-bit mask value.

    Raises:
        ValueError: If the class is unknown.
    """
    return SEMANTIC_OBJECT_OFFSET + CLASS_NAMES.index(category)


def semantic_labels() -> dict[int, str]:
    """Return the meaning of every semantic-mask value.

    Returns:
        Mapping from mask value to label name.
    """
    labels = {SEMANTIC_SKY: "sky", SEMANTIC_SEA: "sea"}
    labels.update({semantic_value(name): name for name in CLASS_NAMES})
    return labels


def coco_bbox(bbox: Sequence[float]) -> list[float]:
    """Convert an edge box to COCO ``[x, y, width, height]``.

    Args:
        bbox: Box as ``(x_min, y_min, x_max, y_max)``.

    Returns:
        The COCO box.
    """
    x_min, y_min, x_max, y_max = bbox
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def yolo_line(class_index: int, bbox: Sequence[float], width: int, height: int) -> str:
    """Format one object as a YOLO label line.

    Args:
        class_index: Zero-based class index.
        bbox: Box as ``(x_min, y_min, x_max, y_max)`` in pixels.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``"<class> <cx> <cy> <w> <h>"`` with coordinates normalised to [0, 1].
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0 / width
    cy = (y_min + y_max) / 2.0 / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    return f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@dataclass(frozen=True)
class DatasetSummary:
    """Counts reported after building the dataset index.

    Attributes:
        images: Number of images indexed.
        objects: Number of labelled objects.
        horizons: Number of images with a visible horizon.
    """

    images: int
    objects: int
    horizons: int


def iter_frames(output_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield the per-image metadata written by the Blender worker.

    Args:
        output_dir: Dataset root directory.

    Yields:
        Parsed ``meta/*.json`` documents in file-name order.
    """
    for path in sorted((output_dir / META_DIR).glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def labelled_objects(frame: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield the objects of a frame that qualify as training labels.

    Args:
        frame: Per-image metadata.

    Yields:
        Object records whose ``valid`` flag is set.
    """
    yield from (obj for obj in frame["objects"] if obj["valid"])


def build_coco(frames: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a COCO object-detection document.

    Args:
        frames: Per-image metadata.

    Returns:
        The COCO document with ``images``, ``annotations`` and ``categories``.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for image_id, frame in enumerate(frames, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": frame["image"],
                "width": frame["width"],
                "height": frame["height"],
            }
        )
        for obj in labelled_objects(frame):
            box = coco_bbox(obj["bbox"])
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CLASS_NAMES.index(obj["category"]) + 1,
                    "bbox": box,
                    "area": obj["visible_pixels"],
                    "iscrowd": 0,
                    "distance_m": obj["distance_m"],
                    "truncated": obj["truncated"],
                }
            )
    categories = [
        {"id": i, "name": name, "supercategory": "maritime"}
        for i, name in enumerate(CLASS_NAMES, start=1)
    ]
    return {
        "info": {"description": "sea-data synthetic maritime dataset"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def horizon_row(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the horizon annotation of a frame into a CSV row.

    Args:
        frame: Per-image metadata.

    Returns:
        A mapping keyed by :data:`HORIZON_COLUMNS`.
    """
    horizon = frame["horizon"]
    camera = frame["camera"]
    endpoints = horizon.get("endpoints") or ((None, None), (None, None))
    (x0, y0), (x1, y1) = endpoints
    return {
        "image": frame["image"],
        "visible": int(horizon["visible"]),
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "y_left": horizon.get("y_left"),
        "y_right": horizon.get("y_right"),
        "angle_deg": horizon.get("angle_deg"),
        "offset_px": horizon.get("offset_px"),
        "dip_deg": horizon["dip_deg"],
        "camera_height_m": camera["height_m"],
        "camera_pitch_deg": camera["pitch_deg"],
        "camera_roll_deg": camera["roll_deg"],
        "transmittance": horizon.get("transmittance"),
    }


def build_index(output_dir: Path) -> DatasetSummary:
    """Write dataset-level annotation files from the per-image metadata.

    Creates ``annotations_coco.json``, YOLO ``labels/*.txt`` with a
    ``dataset.yaml``, and ``horizon.csv`` inside ``output_dir``.

    Args:
        output_dir: Dataset root directory containing ``meta/``.

    Returns:
        Counts of indexed images, labelled objects and visible horizons.
    """
    frames = list(iter_frames(output_dir))
    coco = build_coco(frames)
    (output_dir / COCO_FILE).write_text(json.dumps(coco, indent=1), encoding="utf-8")

    label_dir = output_dir / LABEL_DIR
    label_dir.mkdir(exist_ok=True)
    for frame in frames:
        lines = [
            yolo_line(
                CLASS_NAMES.index(obj["category"]),
                obj["bbox"],
                frame["width"],
                frame["height"],
            )
            for obj in labelled_objects(frame)
        ]
        label_path = label_dir / f"{Path(frame['image']).stem}.txt"
        label_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

    names = "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    (output_dir / YOLO_DATASET_FILE).write_text(
        f"path: {output_dir.resolve()}\ntrain: {IMAGE_DIR}\nval: {IMAGE_DIR}\n"
        f"names:\n{names}",
        encoding="utf-8",
    )

    with (output_dir / HORIZON_FILE).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HORIZON_COLUMNS)
        writer.writeheader()
        writer.writerows(horizon_row(frame) for frame in frames)

    return DatasetSummary(
        images=len(frames),
        objects=len(coco["annotations"]),
        horizons=sum(1 for frame in frames if frame["horizon"]["visible"]),
    )
