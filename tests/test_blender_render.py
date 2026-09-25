"""End-to-end render with a real Blender installation (skipped without one)."""

import json
import shutil
from pathlib import Path

import pytest

from sea_data import cli

pytestmark = [
    pytest.mark.blender,
    pytest.mark.skipif(shutil.which("blender") is None, reason="Blender not installed"),
]


def test_generate_renders_images_masks_and_ground_truth(tmp_path: Path) -> None:
    config = tmp_path / "small.toml"
    config.write_text(
        """
objects_per_image = [2, 2]

[render]
width = 160
height = 90
samples = 4
denoise = false

[weather.fog]
weight = 0.0

[platforms.drone]
weight = 0.0
""",
        encoding="utf-8",
    )
    output = tmp_path / "data"
    code = cli.main(["generate", "-o", str(output), "-n", "1", "-c", str(config)])
    assert code == 0

    meta = json.loads((output / "meta" / "000000.json").read_text())
    assert (output / meta["image"]).stat().st_size > 0
    for key in ("semantic_mask", "instance_mask"):
        assert (output / meta[key]).read_bytes()[:4] == b"\x89PNG"
    assert len(meta["objects"]) == 2

    horizon = meta["horizon"]
    if horizon["visible"] and horizon["mask_error_px"] is not None:
        # The analytic horizon must match the rendered sea/sky boundary.
        assert horizon["mask_error_px"] < 1.5
    for obj in meta["objects"]:
        if obj["bbox"] and obj["amodal_bbox"]:
            x0, y0, x1, y1 = obj["bbox"]
            ax0, ay0, ax1, ay1 = obj["amodal_bbox"]
            assert ax0 - 1 <= x0 and ay0 - 1 <= y0 and x1 <= ax1 + 1 and y1 <= ay1 + 1

    coco = json.loads((output / "annotations_coco.json").read_text())
    assert len(coco["images"]) == 1
    assert (output / "labels" / "000000.txt").exists()
