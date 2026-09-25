"""Verify that a rendered dataset's annotations are internally consistent.

Cross-checks the per-image ``meta/*.json`` files (the source of truth) against
the derived files :func:`sea_data.annotations.build_index` writes -
``annotations_coco.json``, ``labels/*.txt`` and ``horizon.csv`` - and against
the image and mask files on disk. Nothing here depends on Blender.
"""

import csv
import json
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sea_data.annotations import (
    COCO_FILE,
    HORIZON_FILE,
    LABEL_DIR,
    META_DIR,
    coco_bbox,
    iter_frames,
    labelled_objects,
    yolo_line,
)
from sea_data.config import (
    CLASS_NAMES,
    AnnotationConfig,
    ConfigError,
    config_from_mapping,
)


@dataclass(frozen=True)
class Issue:
    """One annotation-correctness problem found in a dataset.

    Attributes:
        frame: Image stem or file the problem relates to, or ``"dataset"`` for
            a dataset-wide problem.
        message: Human-readable description of the problem.
    """

    frame: str
    message: str

    def __str__(self) -> str:
        return f"{self.frame}: {self.message}"


def _png_size(path: Path) -> tuple[int, int] | None:
    """Read the pixel dimensions of a PNG file without decoding it.

    Args:
        path: PNG file.

    Returns:
        ``(width, height)``, or ``None`` if the file cannot be read as PNG.
    """
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    if blob[:8] != b"\x89PNG\r\n\x1a\n" or blob[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", blob[16:24])
    return width, height


def _load_annotation_config(output_dir: Path) -> AnnotationConfig:
    config_path = output_dir / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AnnotationConfig()
    try:
        return config_from_mapping(data).annotation
    except ConfigError:
        return AnnotationConfig()


def verify_files(output_dir: Path, frame: Mapping[str, Any]) -> Iterator[Issue]:
    """Check that the image and mask files a frame references exist and match.

    Args:
        output_dir: Dataset root directory.
        frame: Per-image metadata.

    Yields:
        Issues found.
    """
    stem = frame["image"]
    for key in ("image", "semantic_mask", "instance_mask"):
        try:
            rel_path = frame[key]
        except KeyError:
            yield Issue(stem, f"metadata is missing {key!r}")
            continue
        path = output_dir / rel_path
        if not path.is_file():
            yield Issue(stem, f"{key} file is missing: {rel_path}")

    size = _png_size(output_dir / frame.get("semantic_mask", ""))
    if size is not None and size != (frame["width"], frame["height"]):
        yield Issue(
            stem,
            f"semantic mask size {size} != metadata {(frame['width'], frame['height'])}",
        )


def verify_objects(
    frame: Mapping[str, Any], annotation: AnnotationConfig
) -> Iterator[Issue]:
    """Check the labelled objects of a frame for internal consistency.

    Args:
        frame: Per-image metadata.
        annotation: Thresholds used to decide whether an object is labelled.

    Yields:
        Issues found.
    """
    stem = frame["image"]
    width, height = frame["width"], frame["height"]
    for obj in frame["objects"]:
        tag = f"{stem} instance {obj.get('instance_id')}"
        category = obj.get("category")
        if category not in CLASS_NAMES:
            yield Issue(tag, f"unknown category {category!r}")
        elif obj.get("class_index") != CLASS_NAMES.index(category):
            yield Issue(
                tag, f"class_index {obj.get('class_index')} != index of {category!r}"
            )

        box = obj.get("bbox")
        expected_valid = bool(
            box
            and obj.get("visible_pixels", 0) >= annotation.min_visible_pixels
            and obj.get("transmittance", 0.0) >= annotation.min_transmittance
        )
        if obj.get("valid") != expected_valid:
            yield Issue(tag, f"valid={obj.get('valid')} but should be {expected_valid}")

        if not box:
            continue
        x_min, y_min, x_max, y_max = box
        if not (x_min < x_max and y_min < y_max):
            yield Issue(tag, f"bbox is degenerate: {box}")
        if not (0 <= x_min and x_max <= width and 0 <= y_min and y_max <= height):
            yield Issue(tag, f"bbox {box} exceeds image bounds {(width, height)}")


def verify_horizon(frame: Mapping[str, Any]) -> Iterator[Issue]:
    """Check that a frame's horizon annotation is self-consistent.

    Args:
        frame: Per-image metadata.

    Yields:
        Issues found.
    """
    stem = frame["image"]
    horizon = frame["horizon"]
    endpoints = horizon.get("endpoints")
    if horizon["visible"] and endpoints is None:
        yield Issue(stem, "horizon marked visible but has no endpoints")
    if not horizon["visible"] and endpoints is not None:
        yield Issue(stem, "horizon marked hidden but has endpoints")


def verify_coco(
    output_dir: Path, frames: Sequence[Mapping[str, Any]]
) -> Iterator[Issue]:
    """Check ``annotations_coco.json`` against the frames it was built from.

    Args:
        output_dir: Dataset root directory.
        frames: Per-image metadata, in the order used to build the index.

    Yields:
        Issues found.
    """
    path = output_dir / COCO_FILE
    try:
        coco = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        yield Issue("dataset", f"cannot read {COCO_FILE}: {exc}")
        return

    if len(coco["images"]) != len(frames):
        yield Issue(
            "dataset",
            f"{COCO_FILE} has {len(coco['images'])} images, expected {len(frames)}",
        )
    for image, frame in zip(coco["images"], frames, strict=False):
        stem = frame["image"]
        if image["file_name"] != frame["image"]:
            yield Issue(
                stem, f"coco file_name {image['file_name']!r} != {frame['image']!r}"
            )
        if (image["width"], image["height"]) != (frame["width"], frame["height"]):
            yield Issue(stem, "coco image size does not match metadata")

    image_by_id = {image["id"]: image["file_name"] for image in coco["images"]}
    expected = [
        (image_id, CLASS_NAMES.index(obj["category"]) + 1, coco_bbox(obj["bbox"]))
        for image_id, frame in enumerate(frames, start=1)
        for obj in labelled_objects(frame)
    ]
    actual = [(a["image_id"], a["category_id"], a["bbox"]) for a in coco["annotations"]]
    if actual != expected:
        yield Issue("dataset", f"{COCO_FILE} annotations do not match metadata")
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in image_by_id:
            yield Issue(
                "dataset", f"annotation {annotation['id']} references unknown image_id"
            )
        if not 1 <= annotation["category_id"] <= len(CLASS_NAMES):
            yield Issue(
                "dataset", f"annotation {annotation['id']} has invalid category_id"
            )


def verify_yolo(
    output_dir: Path, frames: Sequence[Mapping[str, Any]]
) -> Iterator[Issue]:
    """Check the YOLO label files against the frames they were built from.

    Args:
        output_dir: Dataset root directory.
        frames: Per-image metadata.

    Yields:
        Issues found.
    """
    for frame in frames:
        stem = frame["image"]
        label_path = output_dir / LABEL_DIR / f"{Path(frame['image']).stem}.txt"
        try:
            text = label_path.read_text(encoding="utf-8")
        except OSError:
            yield Issue(stem, f"YOLO label file is missing: {label_path}")
            continue
        lines = [line for line in text.splitlines() if line]
        expected = [
            yolo_line(
                CLASS_NAMES.index(obj["category"]),
                obj["bbox"],
                frame["width"],
                frame["height"],
            )
            for obj in labelled_objects(frame)
        ]
        if lines != expected:
            yield Issue(stem, "YOLO label file does not match metadata")


def verify_horizon_csv(
    output_dir: Path, frames: Sequence[Mapping[str, Any]]
) -> Iterator[Issue]:
    """Check ``horizon.csv`` against the frames it was built from.

    Args:
        output_dir: Dataset root directory.
        frames: Per-image metadata.

    Yields:
        Issues found.
    """
    path = output_dir / HORIZON_FILE
    try:
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        yield Issue("dataset", f"cannot read {HORIZON_FILE}: {exc}")
        return

    if len(rows) != len(frames):
        yield Issue(
            "dataset", f"{HORIZON_FILE} has {len(rows)} rows, expected {len(frames)}"
        )
    for row, frame in zip(rows, frames, strict=False):
        stem = frame["image"]
        if row["image"] != frame["image"]:
            yield Issue(
                stem, f"horizon.csv image {row['image']!r} != {frame['image']!r}"
            )
        if bool(int(row["visible"])) != bool(frame["horizon"]["visible"]):
            yield Issue(stem, "horizon.csv visible flag does not match metadata")


def verify_dataset(output_dir: Path) -> list[Issue]:
    """Verify every annotation correctness rule for a rendered dataset.

    Args:
        output_dir: Dataset root directory containing ``meta/``.

    Returns:
        All issues found, in no particular order. Empty means the dataset is
        internally consistent.
    """
    if not (output_dir / META_DIR).is_dir():
        return [Issue("dataset", f"no {META_DIR}/ directory in {output_dir}")]

    frames = list(iter_frames(output_dir))
    if not frames:
        return [Issue("dataset", f"no metadata files found in {output_dir / META_DIR}")]

    annotation = _load_annotation_config(output_dir)
    issues: list[Issue] = []
    for frame in frames:
        issues.extend(verify_files(output_dir, frame))
        issues.extend(verify_objects(frame, annotation))
        issues.extend(verify_horizon(frame))
    issues.extend(verify_coco(output_dir, frames))
    issues.extend(verify_yolo(output_dir, frames))
    issues.extend(verify_horizon_csv(output_dir, frames))
    return issues
