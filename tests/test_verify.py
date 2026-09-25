import csv
import json
from pathlib import Path
from typing import Any

from sea_data.annotations import HORIZON_COLUMNS, build_index, write_png
from sea_data.verify import verify_dataset


def _object(category: str = "motorboat", valid: bool = True) -> dict[str, Any]:
    return {
        "instance_id": 1,
        "category": category,
        "class_index": 3,
        "bbox": [10, 20, 30, 60],
        "visible_pixels": 500,
        "distance_m": 120.0,
        "truncated": False,
        "transmittance": 0.5,
        "valid": valid,
    }


def _frame(
    stem: str, objects: list[dict[str, Any]], visible: bool = True
) -> dict[str, Any]:
    return {
        "image": f"images/{stem}.png",
        "width": 40,
        "height": 80,
        "semantic_mask": f"masks/semantic/{stem}.png",
        "instance_mask": f"masks/instance/{stem}.png",
        "camera": {"height_m": 10.0, "pitch_deg": -1.0, "roll_deg": 2.0},
        "horizon": {
            "visible": visible,
            "dip_deg": 0.1,
            "endpoints": [[0.0, 4.0], [20.0, 5.0]] if visible else None,
            "y_left": 4.0 if visible else None,
            "y_right": 5.0 if visible else None,
            "angle_deg": -1.4 if visible else None,
            "offset_px": -7.5 if visible else None,
            "transmittance": 0.5,
        },
        "objects": objects,
    }


def _write_dataset(tmp_path: Path, frames: list[dict[str, Any]]) -> Path:
    meta = tmp_path / "meta"
    meta.mkdir()
    for frame in frames:
        stem = Path(frame["image"]).stem
        (meta / f"{stem}.json").write_text(json.dumps(frame), encoding="utf-8")
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / "masks" / "semantic").mkdir(parents=True, exist_ok=True)
        (tmp_path / "masks" / "instance").mkdir(parents=True, exist_ok=True)
        width, height = frame["width"], frame["height"]
        write_png(
            tmp_path / "images" / f"{stem}.png",
            width,
            height,
            b"\x00" * width * height * 3,
            3,
        )
        write_png(
            tmp_path / frame["semantic_mask"], width, height, b"\x00" * width * height
        )
        write_png(
            tmp_path / frame["instance_mask"], width, height, b"\x00" * width * height
        )
    build_index(tmp_path)
    return tmp_path


def test_verify_clean_dataset_has_no_issues(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, [_frame("000000", [_object()])])
    assert verify_dataset(dataset) == []


def test_verify_reports_missing_meta_dir(tmp_path: Path) -> None:
    issues = verify_dataset(tmp_path)
    assert len(issues) == 1
    assert "meta" in issues[0].message


def test_verify_reports_missing_image_file(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, [_frame("000000", [_object()])])
    (dataset / "images" / "000000.png").unlink()
    issues = verify_dataset(dataset)
    assert any("image file is missing" in str(i) for i in issues)


def test_verify_reports_bbox_out_of_bounds(tmp_path: Path) -> None:
    obj = _object()
    obj["bbox"] = [10, 20, 30, 600]
    dataset = _write_dataset(tmp_path, [_frame("000000", [obj])])
    issues = verify_dataset(dataset)
    assert any("exceeds image bounds" in str(i) for i in issues)


def test_verify_reports_wrong_valid_flag(tmp_path: Path) -> None:
    obj = _object()
    obj["visible_pixels"] = 1
    dataset = _write_dataset(tmp_path, [_frame("000000", [obj])])
    issues = verify_dataset(dataset)
    assert any("but should be False" in str(i) for i in issues)


def test_verify_reports_unknown_category(tmp_path: Path) -> None:
    obj = _object(category="whale", valid=False)
    dataset = _write_dataset(tmp_path, [_frame("000000", [obj])])
    issues = verify_dataset(dataset)
    assert any("unknown category" in str(i) for i in issues)


def test_verify_reports_horizon_inconsistency(tmp_path: Path) -> None:
    frame = _frame("000000", [], visible=True)
    frame["horizon"]["endpoints"] = None
    dataset = _write_dataset(tmp_path, [frame])
    issues = verify_dataset(dataset)
    assert any("no endpoints" in str(i) for i in issues)


def test_verify_reports_tampered_coco_file(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, [_frame("000000", [_object()])])
    coco_path = dataset / "annotations_coco.json"
    coco = json.loads(coco_path.read_text())
    coco["annotations"][0]["bbox"] = [0, 0, 1, 1]
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    issues = verify_dataset(dataset)
    assert any("annotations do not match" in str(i) for i in issues)


def test_verify_reports_tampered_yolo_file(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, [_frame("000000", [_object()])])
    (dataset / "labels" / "000000.txt").write_text(
        "9 0.1 0.1 0.1 0.1\n", encoding="utf-8"
    )
    issues = verify_dataset(dataset)
    assert any("does not match metadata" in str(i) for i in issues)


def test_verify_reports_tampered_horizon_csv(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path, [_frame("000000", [], visible=False)])
    csv_path = dataset / "horizon.csv"
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["visible"] = "1"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HORIZON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    issues = verify_dataset(dataset)
    assert any("visible flag does not match" in str(i) for i in issues)
