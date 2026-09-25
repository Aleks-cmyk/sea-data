import csv
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from sea_data.annotations import (
    build_coco,
    build_index,
    coco_bbox,
    horizon_row,
    semantic_labels,
    semantic_value,
    write_png,
    yolo_line,
)


def _read_png(path: Path) -> tuple[int, int, int, bytes]:
    blob = path.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, header = 8, b"", b""
    while pos < len(blob):
        (length,) = struct.unpack(">I", blob[pos : pos + 4])
        tag = blob[pos + 4 : pos + 8]
        payload = blob[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", blob[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + payload)
        if tag == b"IHDR":
            header = payload
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + length
    width, height, depth, color_type = struct.unpack(">IIBB", header[:10])
    assert depth == 8
    raw = zlib.decompress(idat)
    channels = {0: 1, 2: 3, 6: 4}[color_type]
    stride = width * channels
    rows = [raw[r * (stride + 1) + 1 : (r + 1) * (stride + 1)] for r in range(height)]
    return width, height, channels, b"".join(rows)


def _frame(
    stem: str, objects: list[dict[str, Any]], visible: bool = True
) -> dict[str, Any]:
    return {
        "image": f"images/{stem}.png",
        "width": 200,
        "height": 100,
        "camera": {"height_m": 10.0, "pitch_deg": -1.0, "roll_deg": 2.0},
        "horizon": {
            "visible": visible,
            "dip_deg": 0.1,
            "endpoints": [[0.0, 40.0], [200.0, 45.0]] if visible else None,
            "y_left": 40.0 if visible else None,
            "y_right": 45.0 if visible else None,
            "angle_deg": -1.4 if visible else None,
            "offset_px": -7.5 if visible else None,
            "transmittance": 0.5,
        },
        "objects": objects,
    }


def _object(category: str, valid: bool = True) -> dict[str, Any]:
    return {
        "category": category,
        "bbox": [10, 20, 30, 60],
        "visible_pixels": 500,
        "distance_m": 120.0,
        "truncated": False,
        "valid": valid,
    }


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_write_png_round_trip(tmp_path: Path, channels: int) -> None:
    data = bytes((7 * i) % 256 for i in range(4 * 3 * channels))
    path = tmp_path / "x.png"
    write_png(path, 4, 3, data, channels)
    assert _read_png(path) == (4, 3, channels, data)


def test_write_png_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_png(tmp_path / "x.png", 4, 3, b"\x00" * 5)
    with pytest.raises(ValueError):
        write_png(tmp_path / "x.png", 1, 1, b"\x00\x00", channels=2)


def test_box_formats() -> None:
    assert coco_bbox([10, 20, 30, 60]) == [10, 20, 20, 40]
    assert (
        yolo_line(2, [10, 20, 30, 60], 200, 100)
        == "2 0.100000 0.400000 0.100000 0.400000"
    )


def test_semantic_values() -> None:
    labels = semantic_labels()
    assert labels[0] == "sky" and labels[1] == "sea"
    assert labels[semantic_value("buoy")] == "buoy"
    with pytest.raises(ValueError):
        semantic_value("whale")


def test_build_coco_skips_invalid_objects() -> None:
    coco = build_coco(
        [
            _frame("000000", [_object("sailboat"), _object("buoy", valid=False)]),
            _frame("000001", [_object("cargo_ship")]),
        ]
    )
    assert [img["id"] for img in coco["images"]] == [1, 2]
    assert [(a["image_id"], a["category_id"]) for a in coco["annotations"]] == [
        (1, 3),
        (2, 1),
    ]
    assert coco["annotations"][0]["bbox"] == [10, 20, 20, 40]
    assert coco["categories"][0]["name"] == "cargo_ship"


def test_horizon_row_handles_hidden_horizon() -> None:
    row = horizon_row(_frame("000000", [], visible=False))
    assert row["visible"] == 0
    assert row["x0"] is None and row["y_left"] is None


def test_build_index_writes_all_formats(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    frames = [
        _frame("000000", [_object("motorboat"), _object("buoy", valid=False)]),
        _frame("000001", [], visible=False),
    ]
    for frame in frames:
        stem = Path(frame["image"]).stem
        (meta / f"{stem}.json").write_text(json.dumps(frame), encoding="utf-8")

    summary = build_index(tmp_path)

    assert (summary.images, summary.objects, summary.horizons) == (2, 1, 1)
    coco = json.loads((tmp_path / "annotations_coco.json").read_text())
    assert len(coco["annotations"]) == 1
    assert (tmp_path / "labels" / "000000.txt").read_text() == (
        "3 0.100000 0.400000 0.100000 0.400000\n"
    )
    assert (tmp_path / "labels" / "000001.txt").read_text() == ""
    assert "3: motorboat" in (tmp_path / "dataset.yaml").read_text()
    with (tmp_path / "horizon.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["visible"] for row in rows] == ["1", "0"]
    assert float(rows[0]["y_right"]) == 45.0
