# sea-data

Synthetic training data for **maritime object detection** and **horizon
detection**, rendered with Blender's Python API.

Every image is generated from a randomised but fully reproducible scenario:
time of day, weather (clear, hazy, overcast, fog), sea state (Beaufort 0–8),
camera platform (small boat, ship bridge, shore tower, buoy, drone) and 0–6
procedurally built objects (cargo ships, fishing boats, sailboats,
motorboats, navigation buoys). Each image comes with exact ground truth:

| Output | Content |
| --- | --- |
| `images/<id>.png` | Path-traced render (Cycles, or EEVEE) |
| `masks/semantic/<id>.png` | 0 = sky, 1 = sea, 2… = object class (`2 + class index`) |
| `masks/instance/<id>.png` | 0 = background, 1–254 = object instance |
| `meta/<id>.json` | Boxes, amodal boxes, horizon line, camera intrinsics/pose, full scenario |
| `annotations_coco.json` | COCO detection annotations |
| `labels/<id>.txt`, `dataset.yaml` | YOLO labels and Ultralytics dataset file |
| `horizon.csv` | Horizon line per image (end points, `y_left`/`y_right`, angle, offset) |

## Requirements

- Blender 5.x on `PATH` (or pass `--blender` / set `SEA_DATA_BLENDER`).
  Blender's bundled Python needs `numpy` (included in official builds).
- [uv](https://docs.astral.sh/uv/) for the project itself (Python 3.14).

## Usage

```bash
uv sync

# Preview what would be rendered (no Blender needed)
uv run sea-data sample -n 3 --seed 42

# Render 100 images at 1280x720 with two Blender processes
uv run sea-data generate -o data/run1 -n 100 --seed 42 -j 2

# Faster previews
uv run sea-data generate -o data/preview -n 10 --width 640 --height 360 --samples 16

# GPU / EEVEE
uv run sea-data generate -o data/gpu -n 1000 --device GPU
uv run sea-data generate -o data/eevee -n 1000 --engine BLENDER_EEVEE

# Rebuild COCO/YOLO/horizon files after deleting or editing meta files
uv run sea-data index data/run1
```

Existing images are skipped, so an interrupted run resumes when re-run
(`--overwrite` forces re-rendering). `--start` extends a dataset with new
indices; `(seed, index)` always yields the same scenario.

## Configuration

All randomisation ranges live in `sea_data.config`. Override any subset with
a TOML file passed via `-c`:

```toml
objects_per_image = [1, 4]

[render]
width = 1920
height = 1080
samples = 96

[annotation]
min_visible_pixels = 20     # smaller objects are not labelled
min_transmittance = 0.05    # objects hidden by fog are not labelled

[weather.fog]
weight = 0.3                # sampling weight; 0 disables a preset

[weather.storm]             # new presets must define every field
weight = 0.1
visibility_m = [2000.0, 8000.0]
cloud_cover = [0.9, 1.0]
aerosol_density = [3.0, 8.0]

[platforms.drone]
height_m = [50.0, 200.0]

[object_classes.buoy]
length_m = [2.0, 6.0]
```

The configuration used for a run is saved as `config.json` in the output.

## Ground-truth conventions

- **Image coordinates:** origin top-left, `x` right, `y` down, boxes as
  `[x_min, y_min, x_max, y_max]` in pixel edges (COCO uses `[x, y, w, h]`).
- **Boxes** come from the instance mask, so they include occlusion by waves,
  other objects and the Earth's curvature. `amodal_bbox` is the unoccluded
  projection. Objects are labelled (`valid`) only when they have enough
  visible pixels and are not hidden by haze or fog.
- **Horizon:** the geometric sea horizon, including its dip below the
  horizontal due to camera height. The sea surface is modelled as
  `z = -r² / 2R`, so the rendered horizon matches the analytic one; each
  frame's `mask_error_px` (typically < 0.5 px) checks this against the
  semantic mask. `mask_agreement` is lower when waves or objects hide the
  horizon, and `transmittance` shows how visible it is through haze.
- **Camera:** world `x` east, `y` north, `z` up; yaw is a compass bearing,
  positive pitch looks up, positive roll dips the camera's right side.
  `meta/<id>.json` stores intrinsics and the camera-to-world matrix.

## How it works

- `scenario.py` samples scenarios (pure Python, unit-tested).
- `cli.py` writes job files and starts Blender workers
  (`blender --background --python src/sea_data/blender_worker.py`).
- `scene_builder.py` builds the physical sky (with sun disc), procedural
  clouds, horizon haze and auto-exposure from a small calibration render. It
  also builds the sea: a curved polar-grid mesh with two FFT Ocean layers
  (swell and wind chop with whitecaps), each scaled to the WMO wave height for
  the wind speed.
- `vessels.py` builds the objects and floats them on the evaluated wave
  surface (heave, pitch and roll).
- `render.py` renders the RGB image and then a flat-shaded Workbench pass that
  encodes instance and class IDs exactly, and derives all annotations.

## Development

```bash
uv run ruff format
uv run ruff check
uv run mypy
uv run pytest          # includes a small real Blender render if Blender is installed
```

## Known limitations

- Objects are procedural low-poly models; they give correct silhouettes and
  proportions but not photographic detail.
- Distant waves are faded into a bump map. Rough seas can show faint radial
  streaks in the middle distance.
- Night scenes, rain and camera noise are not simulated. Sensor effects are
  best added as training-time augmentation.
