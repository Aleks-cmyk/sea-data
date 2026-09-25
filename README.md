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

# Check that annotations are internally consistent (bounds, valid flags,
# COCO/YOLO/horizon.csv vs. meta files); exits 1 and lists issues if not
uv run sea-data verify data/run1

# Draw ground-truth boxes and the horizon line onto the images for a visual
# sanity check; writes to data/run1/debug/ by default
uv run sea-data visualize data/run1 --limit 20

# Interactive GUI: page through a dataset's images with the ground-truth
# overlay and metadata panel (arrow keys to move, "Open..." to switch datasets)
uv run sea-data view data/run1
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
min_visible_pixels = 20            # smaller objects are not labelled
min_transmittance = 0.05           # objects hidden by fog are not labelled
min_horizon_transmittance = 0.08   # horizon blended into fog is marked not visible

[land]
probability = 0.3               # chance a coastline appears (default: 0.65)
height_m = [10.0, 120.0]
width_deg = [20.0, 60.0]        # capped further so it never spans the full view
distance_factor = [0.05, 0.6]   # fraction of horizon distance; low = close shoreline

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

## Checking annotations

- `sea-data verify <dataset>` re-derives `annotations_coco.json`,
  `labels/*.txt` and `horizon.csv` from the `meta/*.json` files and reports
  any mismatch, out-of-bounds box, wrong `valid` flag, unknown category or
  missing image/mask file. Exit code is `1` if any issues are found.
- `sea-data visualize <dataset>` stacks the original photo above a copy with
  the object boxes (category colour, or grey when filtered out as
  `valid: false`), the amodal box in white, and the horizon line in yellow
  drawn on top, plus a side panel listing horizon (`visible`, `transmittance`,
  `mask_error_px`, `mask_agreement`), atmosphere (weather, visibility, cloud
  cover, aerosol density), land (present, bearing, width, height, distance)
  and every object's category/distance/valid flag.
  Saves to `<dataset>/debug/` (override with `--overlay-dir`); open the PNGs
  with any image viewer, or pass `--limit N` to only render the first `N`
  images.
- `sea-data view <dataset>` opens a small Tk GUI showing the same
  original/overlay/metadata image, with Prev/Next buttons (or the left/right
  arrow keys) to page through the dataset and an "Open..." button to switch
  datasets.

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
  semantic mask. `mask_agreement` is lower when waves, objects or land hide
  the horizon, and `transmittance` shows how visible it is through haze.
  `visible` is `false` whenever the horizon falls outside the image, only
  grazes a corner, or is fogged in enough that a human could not point to it
  (`transmittance` below `annotation.min_horizon_transmittance`) — never
  `true` for a horizon that cannot actually be seen.
- **Camera:** world `x` east, `y` north, `z` up; yaw is a compass bearing,
  positive pitch looks up, positive roll dips the camera's right side.
  `meta/<id>.json` stores intrinsics and the camera-to-world matrix.

## How it works

- `scenario.py` samples scenarios (pure Python, unit-tested).
- `cli.py` writes job files and starts Blender workers
  (`blender --background --python src/sea_data/blender_worker.py`).
- `scene_builder.py` builds the physical sky (with sun disc), procedural
  clouds (sometimes crisper-edged, see `CLOUD_SHARP_PROBABILITY`), horizon
  haze and auto-exposure from a small calibration render. It also builds the
  sea: a curved polar-grid mesh with two FFT Ocean layers (swell and wind
  chop with whitecaps), each scaled to the WMO wave height for the wind
  speed, with an optional ridged coastline baked into the mesh (`land` in
  the config; never spans the full horizon width, and can sit anywhere from
  right off the bow to the horizon). Waves, foam and ripples are suppressed
  over land, which gets its own mottled earthy material instead of water's.
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

## License

BSD 2-Clause License, see [LICENSE](LICENSE).
