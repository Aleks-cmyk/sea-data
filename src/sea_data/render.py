"""Render one scenario in Blender and derive its ground truth.

Each scenario produces:

* ``images/<stem>.png``: the photo-realistic render.
* ``masks/semantic/<stem>.png``: 0 = sky, 1 = sea, 2+ = object classes.
* ``masks/instance/<stem>.png``: 0 = background, 1-254 = object instances.
* ``meta/<stem>.json``: boxes, horizon line, camera and scenario parameters.

Masks come from a second, flat-shaded Workbench render of the same geometry,
so they include occlusion by waves, other objects and the Earth's curvature.

This module must run inside Blender (it imports ``bpy``).
"""

import argparse
import json
import math
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from sea_data.annotations import (
    IMAGE_DIR,
    MASK_DIR,
    META_DIR,
    semantic_labels,
    semantic_value,
    write_png,
)
from sea_data.config import (
    CLASS_NAMES,
    SEMANTIC_SEA,
    SEMANTIC_SKY,
    AnnotationConfig,
    RenderConfig,
)
from sea_data.geometry import PinholeCamera, horizon_line
from sea_data.scenario import Scenario
from sea_data.scene_builder import (
    Lighting,
    SeaGrid,
    build_camera,
    build_haze_group,
    build_sea,
    build_world,
    calibrate_lighting,
    configure_render,
    read_image_pixels,
    reset_scene,
    sea_extent,
)
from sea_data.vessels import SceneObject, build_objects

type Box = tuple[int, int, int, int]


def render_masks(
    scene: Any, sea: SeaGrid, objects: Sequence[SceneObject], path: Path
) -> tuple[Any, Any]:
    """Render exact semantic and instance masks with the Workbench engine.

    Every object is drawn in a flat, unlit colour whose red channel encodes
    the instance id and whose green channel encodes the semantic class.

    Args:
        scene: The fully built scene.
        sea: The sea grid.
        objects: The placed objects.
        path: Temporary EXR file to render into.

    Returns:
        ``(semantic, instance)`` ``uint8`` arrays of shape ``(height, width)``.
    """
    render = scene.render
    render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "OBJECT"
    shading.show_cavity = False
    shading.show_shadows = False
    shading.show_object_outline = False
    shading.show_specular_highlight = False
    shading.show_backface_culling = False
    scene.display.render_aa = "OFF"
    render.dither_intensity = 0.0
    scene.world.color = (SEMANTIC_SKY / 255.0, SEMANTIC_SKY / 255.0, 0.0)
    scene.view_settings.exposure = 0.0
    sea.obj.color = (0.0, SEMANTIC_SEA / 255.0, 0.0, 1.0)
    for item in objects:
        item.obj.color = (
            item.instance_id / 255.0,
            semantic_value(item.placed.category) / 255.0,
            0.0,
            1.0,
        )
    image = render.image_settings
    image.file_format = "OPEN_EXR"
    image.color_mode = "RGB"
    image.color_depth = "32"
    render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    pixels = read_image_pixels(path)
    codes = np.rint(pixels[..., :2] * 255.0).clip(0, 255).astype(np.uint8)
    return codes[..., 1], codes[..., 0]


def mask_box(mask: Any) -> Box | None:
    """Return the tight bounding box of a boolean mask.

    Args:
        mask: Boolean array of shape ``(height, width)``.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` with exclusive maxima, or ``None``
        for an empty mask.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def amodal_box(obj: Any, camera: PinholeCamera) -> tuple[list[float] | None, bool]:
    """Project an object's vertices to get its unoccluded bounding box.

    Args:
        obj: The Blender object.
        camera: The camera model.

    Returns:
        The box clipped to the image (``None`` if nothing projects into it)
        and whether the object extends beyond the image border.
    """
    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    matrix = np.array(obj.matrix_world)
    world = co @ matrix[:3, :3].T + matrix[:3, 3]
    right, up, back = (np.array(axis) for axis in camera.pose.axes())
    rel = world - np.array((0.0, 0.0, camera.pose.height_m))
    depth = -(rel @ back)
    front = depth > 1e-3
    if not front.any():
        return None, False
    cx, cy = camera.principal_point
    f = camera.focal_px
    u = cx + f * (rel[front] @ right) / depth[front]
    v = cy - f * (rel[front] @ up) / depth[front]
    box = (float(u.min()), float(v.min()), float(u.max()), float(v.max()))
    truncated = (
        not front.all()
        or box[0] < 0
        or box[1] < 0
        or box[2] > camera.width
        or box[3] > camera.height
    )
    clipped = [
        min(max(box[0], 0.0), camera.width),
        min(max(box[1], 0.0), camera.height),
        min(max(box[2], 0.0), camera.width),
        min(max(box[3], 0.0), camera.height),
    ]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None, truncated
    return clipped, truncated


def horizon_mask_agreement(
    semantic: Any, endpoints: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[float | None, float]:
    """Compare the analytic horizon with the sky/sea boundary in the mask.

    For every column whose topmost non-sky pixel is sea, the boundary row is
    compared with the line. Columns hidden by objects are skipped.

    Args:
        semantic: Semantic mask.
        endpoints: Horizon line end points in pixels.

    Returns:
        The median absolute deviation in pixels (``None`` if no column could
        be compared) and the fraction of columns agreeing within 2 pixels,
        which is low when waves or objects hide the horizon.
    """
    (x0, y0), (x1, y1) = endpoints
    if abs(x1 - x0) < 1e-6:
        return None, 0.0
    height, width = semantic.shape
    columns = np.arange(width)
    line_y = y0 + (y1 - y0) * (columns + 0.5 - x0) / (x1 - x0)
    in_range = (line_y >= 0) & (line_y < height)
    not_sky = semantic != SEMANTIC_SKY
    has = not_sky.any(axis=0)
    first = np.argmax(not_sky, axis=0)
    first_label = semantic[first, columns]
    usable = in_range & has & (first_label == SEMANTIC_SEA)
    if not usable.any():
        return None, 0.0
    error = np.abs(first[usable] - line_y[usable])
    agreement = float(np.count_nonzero(error <= 2.0)) / max(int(in_range.sum()), 1)
    return float(np.median(error)), agreement


def _write_masks(output_dir: Path, stem: str, semantic: Any, instance: Any) -> None:
    height, width = semantic.shape
    for kind, mask in (("semantic", semantic), ("instance", instance)):
        folder = output_dir / MASK_DIR / kind
        folder.mkdir(parents=True, exist_ok=True)
        write_png(folder / f"{stem}.png", width, height, mask.tobytes())


def _object_record(
    item: SceneObject,
    instance: Any,
    camera: PinholeCamera,
    scenario: Scenario,
    annotation: AnnotationConfig,
) -> dict[str, Any]:
    placed = item.placed
    visible = instance == item.instance_id
    pixels = int(np.count_nonzero(visible))
    box = mask_box(visible)
    amodal, truncated = amodal_box(item.obj, camera)
    transmittance = scenario.atmosphere.transmittance(placed.distance_m)
    return {
        "instance_id": item.instance_id,
        "category": placed.category,
        "class_index": CLASS_NAMES.index(placed.category),
        "bbox": list(box) if box else None,
        "amodal_bbox": amodal,
        "visible_pixels": pixels,
        "truncated": truncated,
        "distance_m": placed.distance_m,
        "bearing_deg": placed.bearing_deg,
        "heading_deg": placed.heading_deg,
        "length_m": placed.length_m,
        "transmittance": transmittance,
        "valid": bool(
            box
            and pixels >= annotation.min_visible_pixels
            and transmittance >= annotation.min_transmittance
        ),
    }


def _horizon_record(
    camera: PinholeCamera, scenario: Scenario, semantic: Any
) -> dict[str, Any]:
    horizon = horizon_line(camera)
    record = asdict(horizon)
    record["transmittance"] = scenario.atmosphere.transmittance(horizon.distance_m)
    record["mask_error_px"], record["mask_agreement"] = (
        horizon_mask_agreement(semantic, horizon.endpoints)
        if horizon.endpoints
        else (None, 0.0)
    )
    return record


def render_scenario(
    scenario: Scenario,
    render: RenderConfig,
    annotation: AnnotationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Build, render and annotate one scenario.

    Args:
        scenario: The scenario to render.
        render: Render settings.
        annotation: Labelling rules.
        output_dir: Dataset root directory.

    Returns:
        The per-image metadata, which is also written to ``meta/<stem>.json``.
    """
    started = time.perf_counter()
    stem = scenario.stem
    camera = PinholeCamera(
        render.width, render.height, scenario.hfov_deg, scenario.camera
    )
    scene = reset_scene()
    configure_render(scene, render, scenario.index)
    handles = build_world(scene, scenario)
    lighting: Lighting = calibrate_lighting(scene, scenario, handles)
    haze = build_haze_group(lighting.fog_color, scenario.atmosphere.visibility_m)
    sea = build_sea(scene, scenario, camera, haze)
    build_camera(scene, camera, 1.2 * sea_extent(scenario))
    objects = build_objects(scene, scenario.objects, sea, haze)

    suffix = ".png" if render.image_format == "PNG" else ".jpg"
    image_rel = f"{IMAGE_DIR}/{stem}{suffix}"
    (output_dir / IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_dir / image_rel)
    bpy.ops.render.render(write_still=True)

    with tempfile.TemporaryDirectory() as tmp:
        semantic, instance = render_masks(scene, sea, objects, Path(tmp) / "mask.exr")
    _write_masks(output_dir, stem, semantic, instance)

    meta = {
        "image": image_rel,
        "width": render.width,
        "height": render.height,
        "semantic_mask": f"{MASK_DIR}/semantic/{stem}.png",
        "instance_mask": f"{MASK_DIR}/instance/{stem}.png",
        "semantic_labels": semantic_labels(),
        "camera": {
            **asdict(scenario.camera),
            "hfov_deg": camera.hfov_deg,
            "vfov_deg": camera.vfov_deg,
            "intrinsics": camera.intrinsics(),
            "matrix_world": camera.pose.matrix_world(),
        },
        "horizon": _horizon_record(camera, scenario, semantic),
        "objects": [
            _object_record(item, instance, camera, scenario, annotation)
            for item in objects
        ],
        "lighting": asdict(lighting),
        "render_seconds": time.perf_counter() - started,
        "scenario": scenario.to_dict(),
    }
    meta_dir = output_dir / META_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def run_job(argv: Sequence[str]) -> int:
    """Render every scenario of a job file (entry point inside Blender).

    Args:
        argv: Command-line arguments, i.e. ``["--job", "<path>"]``.

    Returns:
        Process exit code: 0 on success, 1 if any scenario failed.
    """
    parser = argparse.ArgumentParser(prog="sea-data-worker")
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text(encoding="utf-8"))
    output_dir = Path(job["output_dir"])
    render = RenderConfig(**job["render"])
    annotation = AnnotationConfig(**job["annotation"])
    failures = 0
    scenarios = [Scenario.from_dict(item) for item in job["scenarios"]]
    for number, scenario in enumerate(scenarios, start=1):
        meta_path = output_dir / META_DIR / f"{scenario.stem}.json"
        if meta_path.exists() and not job.get("overwrite", False):
            print(f"[sea-data] {scenario.stem}: exists, skipped", flush=True)
            continue
        try:
            meta = render_scenario(scenario, render, annotation, output_dir)
        except Exception:  # noqa: BLE001 - keep rendering the remaining frames
            failures += 1
            print(f"[sea-data] {scenario.stem}: FAILED", file=sys.stderr, flush=True)
            traceback.print_exc()
            continue
        labelled = sum(1 for obj in meta["objects"] if obj["valid"])
        print(
            f"[sea-data] {scenario.stem} ({number}/{len(scenarios)}): "
            f"{labelled}/{len(meta['objects'])} objects labelled, "
            f"horizon {'visible' if meta['horizon']['visible'] else 'hidden'}, "
            f"{math.ceil(meta['render_seconds'])} s",
            flush=True,
        )
    return 1 if failures else 0
