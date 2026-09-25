import json
from pathlib import Path
from typing import Any

from PIL import Image

from sea_data.annotations import write_png
from sea_data.visualize import (
    PANEL_WIDTH,
    ROW_GAP,
    category_color,
    frame_metadata_lines,
    visualize_dataset,
)


def _object(category: str = "motorboat", valid: bool = True) -> dict[str, Any]:
    return {
        "instance_id": 1,
        "category": category,
        "bbox": [10, 20, 30, 60],
        "amodal_bbox": [5, 15, 35, 65],
        "visible_pixels": 500,
        "distance_m": 120.0,
        "truncated": False,
        "valid": valid,
    }


def _frame(
    stem: str, objects: list[dict[str, Any]], visible: bool = True
) -> dict[str, Any]:
    return {
        "image": f"images/{stem}.png",
        "width": 40,
        "height": 80,
        "horizon": {
            "visible": visible,
            "endpoints": [[0.0, 4.0], [40.0, 6.0]] if visible else None,
            "transmittance": 0.5,
            "mask_error_px": 1.2,
            "mask_agreement": 0.9,
        },
        "objects": objects,
        "scenario": {
            "atmosphere": {
                "weather": "hazy",
                "visibility_m": 8000.0,
                "cloud_cover": 0.2,
                "aerosol_density": 3.0,
            }
        },
    }


def _write_dataset(tmp_path: Path, frame: dict[str, Any]) -> Path:
    meta = tmp_path / "meta"
    meta.mkdir()
    stem = Path(frame["image"]).stem
    (meta / f"{stem}.json").write_text(json.dumps(frame), encoding="utf-8")
    (tmp_path / "images").mkdir()
    width, height = frame["width"], frame["height"]
    write_png(
        tmp_path / "images" / f"{stem}.png",
        width,
        height,
        b"\x80" * width * height * 3,
        3,
    )
    return tmp_path


def test_category_color_is_stable_and_distinct() -> None:
    assert category_color("cargo_ship") == category_color("cargo_ship")
    assert category_color("cargo_ship") != category_color("buoy")


def test_frame_metadata_lines_include_horizon_atmosphere_and_objects() -> None:
    frame = _frame("000000", [_object()])
    lines = frame_metadata_lines(frame)
    text = "\n".join(lines)
    assert "visible: True" in text
    assert "transmittance: 0.5" in text
    assert "mask_error_px: 1.200" in text
    assert "mask_agreement: 0.900" in text
    assert "weather: hazy" in text
    assert "visibility_m: 8000" in text
    assert "motorboat 120m" in text


def test_visualize_dataset_writes_stacked_comparison(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, _frame("000000", [_object()]))

    written = visualize_dataset(dataset)

    assert written == [dataset / "debug" / "000000.png"]
    comparison = Image.open(written[0])
    assert comparison.size == (40 + PANEL_WIDTH, 2 * 80 + ROW_GAP)


def test_visualize_dataset_marks_invalid_objects_grey(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, _frame("000000", [_object(valid=False)]))

    written = visualize_dataset(dataset)

    comparison = Image.open(written[0]).convert("RGB")
    annotated_y = 80 + ROW_GAP + 20
    assert comparison.getpixel((10, annotated_y)) == (140, 140, 140)


def test_visualize_dataset_respects_overlay_dir_and_limit(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (tmp_path / "images").mkdir()
    for stem in ("000000", "000001"):
        frame = _frame(stem, [])
        (meta / f"{stem}.json").write_text(json.dumps(frame), encoding="utf-8")
        write_png(tmp_path / "images" / f"{stem}.png", 40, 80, b"\x80" * 40 * 80 * 3, 3)

    overlay_dir = tmp_path / "out"
    written = visualize_dataset(tmp_path, overlay_dir=overlay_dir, limit=1)

    assert written == [overlay_dir / "000000.png"]
