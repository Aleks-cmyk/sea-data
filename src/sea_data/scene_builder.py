"""Blender scene construction: render settings, sky, haze, sea and camera.

This module must run inside Blender (it imports ``bpy``).

Lighting model:
    The world uses Blender's physical sky texture including the sun disc, so
    the sun, sky and their ratio are physically consistent. A procedural cloud
    layer is blended over the sky (clouds covering the sun disc also remove
    direct sunlight, and a random fraction of scenarios get crisper-edged
    clouds instead of the soft default), and horizon haze is blended in with
    a path length that grows towards the horizon. A tiny panoramic calibration
    render measures the sky brightness to derive the fog colour, cloud colour
    and auto-exposure.

Sea model:
    A single polar-grid mesh centred below the camera covers the camera's
    field of view out to beyond the geometric horizon. Its rest shape follows
    the Earth's curvature (``z = -r^2 / 2R``), optionally raised by a distant
    coastline silhouette that spans only part of the horizon (see
    :func:`_land_offsets`), so the rendered horizon matches
    :func:`sea_data.geometry.horizon_line` where no land is present. An Ocean
    modifier adds FFT waves, and a Geometry Nodes modifier fades them out with
    distance where the mesh becomes too coarse to represent them (further out
    for rougher sea states); a bump map, boosted in the faded region, carries
    the visual detail there instead of an unrealistic flat band.
"""

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix

from sea_data.config import RenderConfig
from sea_data.geometry import EARTH_RADIUS_M, PinholeCamera, horizon_distance
from sea_data.scenario import Land, Scenario, SeaState

HAZE_SCALE_HEIGHT_M = 1_000.0
"""Height of the haze layer used for the sky's horizon brightening."""

CLOUD_SHARP_PROBABILITY = 0.25
"""Chance that a scenario gets crisper-edged clouds instead of the soft default."""

_SKY_WHITE_LEVEL = 0.85
_GPU_BACKENDS = ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL")


def _luminance(rgb: Any) -> float:
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def reset_scene() -> Any:
    """Reset Blender to an empty factory scene.

    Returns:
        The active ``bpy.types.Scene``.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def _enable_gpu() -> bool:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in _GPU_BACKENDS:
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type != "CPU"]
        if devices:
            for device in prefs.devices:
                device.use = device.type != "CPU"
            return True
    return False


def configure_render(scene: Any, render: RenderConfig, seed: int) -> None:
    """Apply render settings to a scene.

    Args:
        scene: The ``bpy.types.Scene`` to configure.
        render: Render settings.
        seed: Sampling seed, so that noise differs between images.
    """
    settings = scene.render
    settings.engine = render.engine
    settings.resolution_x = render.width
    settings.resolution_y = render.height
    settings.resolution_percentage = 100
    settings.pixel_aspect_x = settings.pixel_aspect_y = 1.0
    settings.film_transparent = False
    if render.engine == "CYCLES":
        cycles = scene.cycles
        cycles.samples = render.samples
        cycles.use_adaptive_sampling = True
        cycles.use_denoising = render.denoise
        cycles.seed = seed
        cycles.max_bounces = 6
        cycles.diffuse_bounces = 2
        cycles.glossy_bounces = 3
        cycles.transmission_bounces = 2
        cycles.volume_bounces = 0
        cycles.caustics_reflective = False
        cycles.caustics_refractive = False
        cycles.device = "GPU" if render.device == "GPU" and _enable_gpu() else "CPU"
    else:
        scene.eevee.taa_render_samples = render.samples
        try:
            scene.eevee.use_raytracing = True
        except AttributeError:
            pass
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass  # Colour management runs in fallback mode (e.g. OCIO mismatch).
    image = settings.image_settings
    image.file_format = render.image_format
    image.color_mode = "RGB"
    if render.image_format == "JPEG":
        image.quality = 92
    else:
        image.color_depth = "8"
        image.compression = 15


@dataclass
class Lighting:
    """Result of the sky calibration.

    Attributes:
        fog_color: Linear-RGB radiance of the haze near the horizon.
        cloud_color: Linear-RGB radiance of lit clouds.
        exposure_ev: Exposure applied to the view transform.
        sky_luminance: Median luminance of the clear sky.
    """

    fog_color: tuple[float, float, float]
    cloud_color: tuple[float, float, float]
    exposure_ev: float
    sky_luminance: float


class _NodeBuilder:
    """Small helper to create and link shader nodes tersely."""

    def __init__(self, tree: Any) -> None:
        self.tree = tree

    def node(self, kind: str, **props: Any) -> Any:
        node = self.tree.nodes.new(kind)
        for name, value in props.items():
            setattr(node, name, value)
        return node

    def link(self, output: Any, target: Any) -> None:
        self.tree.links.new(output, target)

    def math(self, operation: str, a: Any, b: Any = None) -> Any:
        node = self.node("ShaderNodeMath", operation=operation)
        for socket, value in zip(node.inputs, (a, b), strict=False):
            if value is None:
                continue
            if isinstance(value, (int, float)):
                socket.default_value = value
            else:
                self.link(value, socket)
        return node.outputs[0]

    def vmath(self, operation: str, a: Any, b: Any = None) -> Any:
        node = self.node("ShaderNodeVectorMath", operation=operation)
        for socket, value in zip(node.inputs, (a, b), strict=False):
            if value is None:
                continue
            if isinstance(value, tuple):
                socket.default_value = value
            else:
                self.link(value, socket)
        return node.outputs["Vector"]

    def map_range(
        self, value: Any, from_min: float, from_max: float, to_min: Any, to_max: Any
    ) -> Any:
        node = self.node("ShaderNodeMapRange", clamp=True)
        self.link(value, node.inputs["Value"])
        node.inputs["From Min"].default_value = from_min
        node.inputs["From Max"].default_value = from_max
        for socket, target in ((to_min, "To Min"), (to_max, "To Max")):
            if isinstance(socket, (int, float)):
                node.inputs[target].default_value = socket
            else:
                self.link(socket, node.inputs[target])
        return node.outputs["Result"]

    def mix_color(self, factor: Any, a: Any, b: Any, blend: str = "MIX") -> Any:
        node = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.link(factor, node.inputs["Factor"])
        for value, index in ((a, 6), (b, 7)):
            if isinstance(value, tuple):
                node.inputs[index].default_value = (*value, 1.0)
            else:
                self.link(value, node.inputs[index])
        return node.outputs[2]


def _sky_texture(nodes: _NodeBuilder, scenario: Scenario) -> Any:
    sky = nodes.node("ShaderNodeTexSky")
    for sky_type in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA"):
        try:
            sky.sky_type = sky_type
            break
        except TypeError:
            continue
    sky.sun_disc = True
    sky.sun_elevation = math.radians(scenario.sun.elevation_deg)
    sky.sun_rotation = math.radians(scenario.sun.azimuth_deg)
    sky.altitude = min(scenario.camera.height_m, 1_000.0)
    sky.aerosol_density = scenario.atmosphere.aerosol_density
    return sky


def build_world(scene: Any, scenario: Scenario) -> dict[str, Any]:
    """Create the world shader: physical sky, clouds and horizon haze.

    The returned handles let :func:`calibrate_lighting` switch clouds and haze
    off during calibration and set the calibrated colours afterwards.

    Args:
        scene: Scene receiving the world.
        scenario: Scenario providing sun, clouds and visibility.

    Returns:
        Handles to the tunable nodes.
    """
    world = bpy.data.worlds.new("Sky")
    scene.world = world
    tree = world.node_tree
    tree.nodes.clear()
    nodes = _NodeBuilder(tree)
    atmosphere = scenario.atmosphere
    cover = atmosphere.cloud_cover

    direction = nodes.vmath(
        "NORMALIZE", nodes.node("ShaderNodeTexCoord").outputs["Generated"]
    )
    z = nodes.node("ShaderNodeSeparateXYZ")
    nodes.link(direction, z.inputs[0])
    elevation = z.outputs["Z"]
    sky = _sky_texture(nodes, scenario)

    # Clouds: noise projected onto a flat layer, thresholded by cloud cover.
    projected = nodes.vmath("DIVIDE", direction, nodes.math("MAXIMUM", elevation, 0.02))
    rng = np.random.default_rng(atmosphere.cloud_seed)
    sharp = bool(rng.random() < CLOUD_SHARP_PROBABILITY)
    offset = tuple(float(v) for v in rng.uniform(-1000.0, 1000.0, size=3))
    layer = nodes.vmath(
        "ADD", nodes.vmath("MULTIPLY", projected, (0.9, 0.9, 0.0)), offset
    )
    noise = nodes.node("ShaderNodeTexNoise", noise_dimensions="3D")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.6
    nodes.link(layer, noise.inputs["Vector"])
    threshold = 0.68 - 0.40 * cover
    edge_soft, edge_spread = (0.015, 0.05) if sharp else (0.06, 0.2)
    density = nodes.map_range(
        noise.outputs["Fac"], threshold - edge_soft, threshold + edge_spread, 0.0, 1.0
    )
    near_horizon = nodes.map_range(elevation, 0.0, 0.08, cover**2, 1.0)
    cloud_switch = nodes.node("ShaderNodeValue", label="cloud_switch")
    cloud_switch.outputs[0].default_value = 1.0
    cloud_mask = nodes.math(
        "MULTIPLY",
        nodes.math("MULTIPLY", density, near_horizon),
        cloud_switch.outputs[0],
    )
    cloud_rgb = nodes.node("ShaderNodeRGB", label="cloud_color")
    shading = nodes.map_range(
        noise.outputs["Fac"], threshold, threshold + 0.35, 1.0, 0.65
    )
    cloud_color = nodes.vmath("SCALE", cloud_rgb.outputs[0])
    nodes.link(shading, cloud_color.node.inputs["Scale"])
    clouded = nodes.mix_color(cloud_mask, sky.outputs["Color"], cloud_color)

    # Haze: optical depth through a layer of HAZE_SCALE_HEIGHT_M.
    depth = 3.912 * HAZE_SCALE_HEIGHT_M / atmosphere.visibility_m
    optical = nodes.math("DIVIDE", depth, nodes.math("MAXIMUM", elevation, 0.0015))
    haze = nodes.math(
        "SUBTRACT", 1.0, nodes.math("EXPONENT", nodes.math("MULTIPLY", optical, -1.0))
    )
    haze_switch = nodes.node("ShaderNodeValue", label="haze_switch")
    haze_switch.outputs[0].default_value = 1.0
    haze = nodes.math("MULTIPLY", haze, haze_switch.outputs[0])
    fog_rgb = nodes.node("ShaderNodeRGB", label="fog_color")
    final = nodes.mix_color(haze, clouded, fog_rgb.outputs[0])

    background = nodes.node("ShaderNodeBackground")
    nodes.link(final, background.inputs["Color"])
    output = nodes.node("ShaderNodeOutputWorld")
    nodes.link(background.outputs[0], output.inputs["Surface"])
    return {
        "cloud_switch": cloud_switch,
        "haze_switch": haze_switch,
        "cloud_color": cloud_rgb,
        "fog_color": fog_rgb,
    }


def _render_panorama(scene: Any, path: Path) -> Any:
    camera_data = bpy.data.cameras.new("Calibration")
    camera_data.type = "PANO"
    camera_data.panorama_type = "EQUIRECTANGULAR"
    camera = bpy.data.objects.new("Calibration", camera_data)
    scene.collection.objects.link(camera)
    camera.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    engine = scene.render.engine
    resolution = scene.render.resolution_x, scene.render.resolution_y
    image = scene.render.image_settings
    stored = image.file_format, image.color_depth, image.color_mode
    samples, denoise = scene.cycles.samples, scene.cycles.use_denoising
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = 64, 32
    scene.cycles.samples, scene.cycles.use_denoising = 16, False
    image.file_format, image.color_depth, image.color_mode = "OPEN_EXR", "32", "RGB"
    scene.render.filepath = str(path)
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        scene.render.engine = engine
        scene.render.resolution_x, scene.render.resolution_y = resolution
        image.file_format, image.color_depth, image.color_mode = stored
        scene.cycles.samples, scene.cycles.use_denoising = samples, denoise
        bpy.data.objects.remove(camera)
    return read_image_pixels(path)


def read_image_pixels(path: Path) -> Any:
    """Load an image file as a float array with the top row first.

    Args:
        path: Image file readable by Blender (EXR for exact values).

    Returns:
        ``numpy`` array of shape ``(height, width, 4)``.
    """
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
    finally:
        bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)[::-1]


def _panorama_elevations(pano: Any) -> Any:
    rows = pano.shape[0]
    elevation = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
    return np.broadcast_to(elevation[:, None], pano.shape[:2])


def calibrate_lighting(
    scene: Any, scenario: Scenario, handles: dict[str, Any]
) -> Lighting:
    """Measure the sky and derive fog colour, cloud colour and exposure.

    Args:
        scene: Scene whose world was created by :func:`build_world`.
        scenario: Scenario providing cloud cover and exposure jitter.
        handles: Node handles returned by :func:`build_world`.

    Returns:
        The calibrated lighting, already applied to the world and the scene.
    """
    handles["cloud_switch"].outputs[0].default_value = 0.0
    handles["haze_switch"].outputs[0].default_value = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        pano = _render_panorama(scene, Path(tmp) / "sky.exr")
    elevation = _panorama_elevations(pano)
    rgb = pano[..., :3].astype(np.float64)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    sky_lum = max(float(np.median(lum[elevation > 5.0])), 1e-6)
    horizon_rows = (elevation > 0.0) & (elevation < 6.0)
    horizon_rgb = np.median(rgb[horizon_rows].reshape(-1, 3), axis=0)
    horizon_lum = max(_luminance(horizon_rgb), 1e-6)

    atmosphere = scenario.atmosphere
    cover = atmosphere.cloud_cover
    grey = np.full(3, horizon_lum)
    desaturate = float(np.clip(1.0 - atmosphere.visibility_m / 20_000.0, 0.3, 0.9))
    fog = (1.0 - desaturate) * horizon_rgb + desaturate * grey
    fog *= 1.0 - 0.55 * cover
    brightness = 1.25 - 0.75 * cover
    cloud = brightness * (0.5 * grey + 0.5 * horizon_rgb)
    handles["cloud_switch"].outputs[0].default_value = 1.0
    handles["haze_switch"].outputs[0].default_value = 1.0
    handles["fog_color"].outputs[0].default_value = (*fog, 1.0)
    handles["cloud_color"].outputs[0].default_value = (*cloud, 1.0)

    # Auto-exposure on the final sky: keep its bright end below clipping.
    with tempfile.TemporaryDirectory() as tmp:
        final = _render_panorama(scene, Path(tmp) / "sky.exr")
    final_lum = final[..., :3].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    bright = float(np.percentile(final_lum[_panorama_elevations(final) > 1.0], 90))
    exposure = math.log2(_SKY_WHITE_LEVEL / max(bright, 1e-6))
    exposure = float(np.clip(exposure + scenario.exposure_offset_ev, -12.0, 14.0))
    scene.view_settings.exposure = exposure
    return Lighting(
        fog_color=(float(fog[0]), float(fog[1]), float(fog[2])),
        cloud_color=(float(cloud[0]), float(cloud[1]), float(cloud[2])),
        exposure_ev=exposure,
        sky_luminance=sky_lum,
    )


def build_haze_group(fog_color: tuple[float, float, float], visibility_m: float) -> Any:
    """Create a shader node group that blends a surface into the haze.

    The blend factor follows Koschmieder's law, ``1 - exp(-3.912 d / V)``,
    using the distance ``d`` from the camera.

    Args:
        fog_color: Linear-RGB radiance of the haze.
        visibility_m: Meteorological visibility in metres.

    Returns:
        The ``ShaderNodeTree`` group with one shader input and output.
    """
    group = bpy.data.node_groups.new("Haze", "ShaderNodeTree")
    group.interface.new_socket("Shader", in_out="INPUT", socket_type="NodeSocketShader")
    group.interface.new_socket(
        "Shader", in_out="OUTPUT", socket_type="NodeSocketShader"
    )
    nodes = _NodeBuilder(group)
    group_in = nodes.node("NodeGroupInput")
    group_out = nodes.node("NodeGroupOutput")
    distance = nodes.node("ShaderNodeCameraData").outputs["View Distance"]
    extinction = nodes.math("MULTIPLY", distance, -3.912 / visibility_m)
    factor = nodes.math("SUBTRACT", 1.0, nodes.math("EXPONENT", extinction))
    emission = nodes.node("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*fog_color, 1.0)
    mix = nodes.node("ShaderNodeMixShader")
    nodes.link(factor, mix.inputs["Fac"])
    nodes.link(group_in.outputs[0], mix.inputs[1])
    nodes.link(emission.outputs[0], mix.inputs[2])
    nodes.link(mix.outputs[0], group_out.inputs[0])
    return group


def new_material(
    name: str,
    haze: Any,
    color: tuple[float, float, float],
    roughness: float = 0.5,
    metallic: float = 0.0,
) -> Any:
    """Create a Principled BSDF material wrapped in the haze group.

    Args:
        name: Material name.
        haze: Haze node group from :func:`build_haze_group`.
        color: Linear-RGB base colour.
        roughness: Surface roughness.
        metallic: Metalness.

    Returns:
        The new ``bpy.types.Material``.
    """
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    nodes = _NodeBuilder(tree)
    bsdf = nodes.node("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    _finish_with_haze(nodes, bsdf.outputs[0], haze)
    return material


def _finish_with_haze(nodes: _NodeBuilder, shader: Any, haze: Any) -> None:
    group = nodes.node("ShaderNodeGroup", node_tree=haze)
    nodes.link(shader, group.inputs[0])
    output = nodes.node("ShaderNodeOutputMaterial")
    nodes.link(group.outputs[0], output.inputs["Surface"])


def sea_extent(scenario: Scenario) -> float:
    """Return the radius the sea mesh must cover.

    Args:
        scenario: The scenario to render.

    Returns:
        Radius in metres, beyond both the geometric horizon and every object.
    """
    furthest = max(
        (obj.distance_m + obj.length_m for obj in scenario.objects), default=0.0
    )
    return max(2.0 * horizon_distance(scenario.camera.height_m), furthest, 2_000.0)


@dataclass
class SeaGrid:
    """The polar grid underlying the sea mesh.

    Attributes:
        obj: The sea ``bpy.types.Object``.
        radii: Ring radii in metres.
        bearings: Column bearings in radians.
        fade_start_m: Range where wave displacement starts to fade.
        fade_end_m: Range beyond which waves are only a bump map.
    """

    obj: Any
    radii: Any
    bearings: Any
    fade_start_m: float
    fade_end_m: float

    def surface_heights(self, points_xy: Any) -> Any:
        """Sample the evaluated (wavy) sea surface height.

        Args:
            points_xy: Array of shape ``(n, 2)`` with world x/y coordinates.

        Returns:
            Array of ``n`` heights; points outside the grid get the curved
            rest height.
        """
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = self.obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
        finally:
            evaluated.to_mesh_clear()
        z = co[2::3].reshape(len(self.radii), len(self.bearings))
        points = np.asarray(points_xy, dtype=np.float64)
        r = np.hypot(points[:, 0], points[:, 1])
        bearing = np.arctan2(points[:, 0], points[:, 1])
        rest = -(r**2) / (2.0 * EARTH_RADIUS_M)
        start = self.bearings[0]
        step = self.bearings[1] - self.bearings[0]
        rel = np.mod(bearing - start, 2.0 * np.pi) / step
        ring = np.interp(r, self.radii, np.arange(len(self.radii)))
        inside = (
            (rel <= len(self.bearings) - 1)
            & (r >= self.radii[0])
            & (r <= self.radii[-1])
        )
        heights = rest.copy()
        i0 = np.clip(np.floor(ring).astype(int), 0, len(self.radii) - 2)
        j0 = np.clip(np.floor(rel).astype(int), 0, len(self.bearings) - 2)
        fi, fj = ring - i0, rel - j0
        bilinear = (
            z[i0, j0] * (1 - fi) * (1 - fj)
            + z[i0 + 1, j0] * fi * (1 - fj)
            + z[i0, j0 + 1] * (1 - fi) * fj
            + z[i0 + 1, j0 + 1] * fi * fj
        )
        heights[inside] = bilinear[inside]
        return heights


def _sector_half_width(camera: PinholeCamera) -> float:
    half_h = math.radians(camera.hfov_deg) / 2.0
    half_v = math.radians(camera.vfov_deg) / 2.0
    cone = math.atan(math.hypot(math.tan(half_h), math.tan(half_v)))
    ratio = math.sin(cone) / max(math.cos(math.radians(camera.pose.pitch_deg)), 1e-6)
    if ratio >= 1.0:
        return math.pi
    return min(math.asin(ratio) + math.radians(12.0), math.pi)


def _polar_grid(camera: PinholeCamera, extent_m: float) -> tuple[Any, Any]:
    pixel_angle = math.radians(camera.hfov_deg) / camera.width
    step = float(np.clip(3.0 * pixel_angle, 0.0008, 0.01))
    half = _sector_half_width(camera)
    columns = math.ceil(2.0 * half / step) + 1
    centre = math.radians(camera.pose.yaw_deg)
    bearings = centre - half + np.arange(columns) * (2.0 * half / (columns - 1))
    radii = [0.4]
    min_cell = float(np.clip(0.12 * camera.pose.height_m, 0.15, 1.0))
    while radii[-1] < extent_m:
        radii.append(radii[-1] + max(min_cell, 1.5 * step * radii[-1]))
    return np.array(radii), bearings


def _grid_mesh(
    name: str,
    radii: Any,
    bearings: Any,
    ring_weights: dict[str, Any],
    height_offset: Any | None = None,
) -> Any:
    rr, bb = np.meshgrid(radii, bearings, indexing="ij")
    z = -(rr**2) / (2.0 * EARTH_RADIUS_M)
    if height_offset is not None:
        z = z + height_offset
    verts = np.stack((rr * np.sin(bb), rr * np.cos(bb), z), axis=-1).reshape(-1, 3)
    rows, cols = rr.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    quads = np.stack(
        (index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]), axis=-1
    ).reshape(-1, 4)

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(quads.size)
    mesh.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    mesh.polygons.add(len(quads))
    mesh.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    mesh.update(calc_edges=True)
    mesh.shade_smooth()
    rest = mesh.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    rest.data.foreach_set("vector", verts.astype(np.float32).ravel())
    for attr_name, weights in ring_weights.items():
        attribute = mesh.attributes.new(attr_name, "FLOAT", "POINT")
        values = np.asarray(weights)
        # A 1D array gives one weight per ring, repeated across every
        # bearing; a 2D (radii x bearings) array is used directly, e.g. for
        # weights that also vary with bearing such as land coverage.
        flat = np.repeat(values, len(bearings)) if values.ndim == 1 else values.ravel()
        attribute.data.foreach_set("value", flat.astype(np.float32))
    return mesh


def _smoothstep(edge0: float, edge1: float, x: Any) -> Any:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _terrain_ridge(bearings: Any, seed: int, harmonics: int = 4) -> Any:
    """Return a multi-harmonic ridge profile along a set of bearings.

    Sums a handful of random sine waves to give a coastline an irregular,
    hilly skyline instead of a flat plateau; roughly in ``[0.2, 1.8]``.

    Args:
        bearings: Column bearings of the sea grid, in radians.
        seed: Seed for the harmonic amplitudes, phases and frequencies.
        harmonics: Number of sine components to sum.

    Returns:
        Array of shape ``(len(bearings),)``.
    """
    rng = np.random.default_rng(seed)
    span = max(float(bearings[-1] - bearings[0]), 1e-6)
    position = (bearings - bearings[0]) / span
    profile = np.ones_like(bearings)
    for k in range(1, harmonics + 1):
        amplitude = rng.uniform(0.12, 0.35) / k
        phase = rng.uniform(0.0, 2.0 * np.pi)
        cycles = k * rng.uniform(1.5, 3.0)
        profile = profile + amplitude * np.sin(2.0 * np.pi * cycles * position + phase)
    return np.clip(profile, 0.2, 1.8)


def _land_offsets(
    radii: Any, bearings: Any, land: Land, horizon_m: float
) -> tuple[Any, Any] | tuple[None, None]:
    """Return a per-vertex height offset and coverage mask for a coastline.

    The silhouette starts at ``land.distance_factor`` times the horizon
    distance (close to the camera for a nearby shoreline, near the horizon
    for a distant one) and tapers to zero at both its angular edges, so it
    never covers the full width of the visible horizon. Its skyline is
    ridged rather than flat (see :func:`_terrain_ridge`).

    Args:
        radii: Ring radii of the sea grid, in metres.
        bearings: Column bearings of the sea grid, in radians.
        land: Land parameters.
        horizon_m: Distance to the geometric horizon, in metres.

    Returns:
        ``(height_offset, coverage)``, each of shape
        ``(len(radii), len(bearings))``, or ``(None, None)`` if no land was
        sampled for this scenario. ``coverage`` is the land footprint before
        the ridge height variation, for suppressing waves and foam.
    """
    if not land.present:
        return None, None
    centre = math.radians(land.bearing_deg)
    half_width = math.radians(land.width_deg) / 2.0
    delta = np.abs(np.mod(bearings - centre + math.pi, 2.0 * math.pi) - math.pi)
    edge = max(0.2 * half_width, 1e-6)
    angular = 1.0 - _smoothstep(half_width - edge, half_width, delta)

    near = land.distance_factor * horizon_m
    far = min(float(radii[-1]) - 1.0, horizon_m)
    depth = max(far - near, 10.0)
    plateau = near + 0.35 * depth
    fall = far - 0.15 * depth
    radial = _smoothstep(near, plateau, radii) * (1.0 - _smoothstep(fall, far, radii))

    coverage = radial[:, None] * angular[None, :]
    ridge = _terrain_ridge(bearings, land.seed)
    height = land.height_m * coverage * ridge[None, :]
    return height, coverage


def _fade_node_group(rest_name: str, fade_name: str, store: str | None = None) -> Any:
    """Blend positions back towards a rest attribute by a per-vertex weight.

    Computes ``rest + fade * (position - rest)`` and optionally stores the
    result as a new attribute, which serves as the rest shape of the next
    wave layer.
    """
    group = bpy.data.node_groups.new(f"Fade.{fade_name}", "GeometryNodeTree")
    group.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    group.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    nodes = _NodeBuilder(group)
    group_in = nodes.node("NodeGroupInput")
    group_out = nodes.node("NodeGroupOutput")
    rest = nodes.node("GeometryNodeInputNamedAttribute", data_type="FLOAT_VECTOR")
    rest.inputs["Name"].default_value = rest_name
    fade = nodes.node("GeometryNodeInputNamedAttribute", data_type="FLOAT")
    fade.inputs["Name"].default_value = fade_name
    position = nodes.node("GeometryNodeInputPosition")
    mix = nodes.node("ShaderNodeMix", data_type="VECTOR")
    nodes.link(fade.outputs["Attribute"], mix.inputs["Factor"])
    nodes.link(rest.outputs["Attribute"], mix.inputs[4])
    nodes.link(position.outputs[0], mix.inputs[5])
    set_position = nodes.node("GeometryNodeSetPosition")
    nodes.link(group_in.outputs[0], set_position.inputs["Geometry"])
    nodes.link(mix.outputs[1], set_position.inputs["Position"])
    geometry = set_position.outputs[0]
    if store is not None:
        stored = nodes.node(
            "GeometryNodeStoreNamedAttribute", data_type="FLOAT_VECTOR", domain="POINT"
        )
        stored.inputs["Name"].default_value = store
        nodes.link(geometry, stored.inputs["Geometry"])
        nodes.link(mix.outputs[1], stored.inputs["Value"])
        geometry = stored.outputs[0]
    nodes.link(geometry, group_out.inputs[0])
    return group


def _sea_material(scenario: Scenario, haze: Any) -> Any:
    sea = scenario.sea
    material = bpy.data.materials.new("Sea")
    tree = material.node_tree
    tree.nodes.clear()
    nodes = _NodeBuilder(tree)

    fade = nodes.node("ShaderNodeAttribute", attribute_type="GEOMETRY")
    fade.attribute_name = "fade"
    foam = nodes.node("ShaderNodeAttribute", attribute_type="GEOMETRY")
    foam.attribute_name = "foam"
    land = nodes.node("ShaderNodeAttribute", attribute_type="GEOMETRY")
    land.attribute_name = "land"
    not_land = nodes.math("SUBTRACT", 1.0, land.outputs["Fac"])
    # The foam layer is an sRGB byte colour, so the shader sees linear values.
    # No whitecaps on land.
    foam_factor = nodes.math(
        "MULTIPLY",
        nodes.math(
            "MULTIPLY",
            nodes.map_range(foam.outputs["Fac"], 0.15, 0.7, 0.0, 1.0),
            fade.outputs["Fac"],
        ),
        not_land,
    )

    coords = nodes.node("ShaderNodeTexCoord").outputs["Object"]
    ripples = nodes.node("ShaderNodeTexNoise", noise_dimensions="3D")
    ripples.inputs["Scale"].default_value = 0.6
    ripples.inputs["Detail"].default_value = 8.0
    ripples.inputs["Roughness"].default_value = 0.55
    nodes.link(nodes.vmath("MULTIPLY", coords, (1.0, 1.0, 0.0)), ripples.inputs[0])
    bump = nodes.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    # Displaced geometry fades to flat towards the horizon (see _wave_fades);
    # boost the bump amplitude there so the surface keeps looking textured
    # instead of turning into an unrealistic flat, calm band. No ripples
    # on land, which isn't water.
    base_distance = 0.02 + 0.05 * sea.beaufort
    fade_boost = nodes.map_range(fade.outputs["Fac"], 0.0, 1.0, 2.5, 1.0)
    nodes.link(
        nodes.math(
            "MULTIPLY", nodes.math("MULTIPLY", base_distance, fade_boost), not_land
        ),
        bump.inputs["Distance"],
    )
    nodes.link(ripples.outputs["Fac"], bump.inputs["Height"])

    # Mottled land colour: two earthy tones mixed by a coarse noise texture.
    land_noise = nodes.node("ShaderNodeTexNoise", noise_dimensions="3D")
    land_noise.inputs["Scale"].default_value = 0.015
    land_noise.inputs["Detail"].default_value = 4.0
    land_noise.inputs["Roughness"].default_value = 0.6
    nodes.link(coords, land_noise.inputs[0])
    land_low = nodes.node("ShaderNodeRGB", label="land_low")
    land_low.outputs[0].default_value = (0.05, 0.07, 0.025, 1.0)
    land_high = nodes.node("ShaderNodeRGB", label="land_high")
    land_high.outputs[0].default_value = (0.14, 0.12, 0.08, 1.0)
    land_color = nodes.mix_color(
        land_noise.outputs["Fac"], land_low.outputs[0], land_high.outputs[0]
    )

    water_rgb = nodes.node("ShaderNodeRGB", label="water_color")
    water_rgb.outputs[0].default_value = (*sea.water_color, 1.0)
    base_color = nodes.mix_color(land.outputs["Fac"], water_rgb.outputs[0], land_color)

    water_roughness = 0.02 + 0.015 * sea.beaufort
    land_roughness = 0.9
    surface_roughness = nodes.math(
        "ADD",
        water_roughness,
        nodes.math("MULTIPLY", land.outputs["Fac"], land_roughness - water_roughness),
    )

    water = nodes.node("ShaderNodeBsdfPrincipled")
    nodes.link(base_color, water.inputs["Base Color"])
    nodes.link(surface_roughness, water.inputs["Roughness"])
    water.inputs["IOR"].default_value = 1.333
    nodes.link(bump.outputs["Normal"], water.inputs["Normal"])
    whitecap = nodes.node("ShaderNodeBsdfPrincipled")
    whitecap.inputs["Base Color"].default_value = (0.75, 0.78, 0.8, 1.0)
    whitecap.inputs["Roughness"].default_value = 0.7
    surface = nodes.node("ShaderNodeMixShader")
    nodes.link(foam_factor, surface.inputs["Fac"])
    nodes.link(water.outputs[0], surface.inputs[1])
    nodes.link(whitecap.outputs[0], surface.inputs[2])
    _finish_with_haze(nodes, surface.outputs[0], haze)
    return material


def _wave_fades(
    radii: Any, wavelength_m: float, step: float, horizon_m: float, reach: float = 1.0
) -> tuple[Any, float, float]:
    """Return per-ring wave weights that fade out where the mesh gets coarse.

    Waves are faded before ``0.7`` times the horizon distance so that the
    horizon silhouette stays exactly on the analytic curve. ``reach`` pushes
    the fade further out for rougher sea states, so a visible calm band does
    not appear well before the horizon while whitecaps churn in the
    foreground.
    """
    end = min(max(wavelength_m / (6.0 * step), 45.0) * reach, 0.7 * horizon_m)
    start = min(max(wavelength_m / (12.0 * step), 30.0) * reach, 0.6 * end)
    fade = np.clip((end - radii) / (end - start), 0.0, 1.0)
    return fade * fade * (3.0 - 2.0 * fade), start, end


def _add_ocean(
    obj: Any,
    name: str,
    sea: SeaState,
    wind_ms: float,
    spatial_size: float,
    choppiness: float,
    seed_offset: int,
) -> Any:
    ocean = obj.modifiers.new(name, "OCEAN")
    ocean.geometry_mode = "DISPLACE"
    ocean.spectrum = "PIERSON_MOSKOWITZ"
    ocean.resolution = ocean.viewport_resolution = 12
    ocean.spatial_size = int(spatial_size)
    ocean.wind_velocity = max(wind_ms, 0.3)
    ocean.wave_direction = math.radians(90.0 - sea.wind_direction_deg)
    ocean.wave_alignment = sea.wave_alignment
    ocean.choppiness = choppiness
    ocean.damping = 0.5
    ocean.depth = 200.0
    ocean.random_seed = sea.ocean_seed + seed_offset
    ocean.time = sea.time_s
    return ocean


def _measure_wave_spread(obj: Any, layer: str, weights: Any) -> float:
    """Measure the vertical displacement spread of one ocean layer alone."""
    oceans = [m for m in obj.modifiers if m.type == "OCEAN"]
    for modifier in oceans:
        modifier.show_viewport = modifier.name == layer
        if modifier.name == layer:
            modifier.wave_scale = 1.0
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
    finally:
        evaluated.to_mesh_clear()
        for modifier in oceans:
            modifier.show_viewport = True
    rest = np.empty_like(co)
    obj.data.attributes["rest"].data.foreach_get("vector", rest)
    near = np.repeat(weights >= 1.0, len(co) // 3 // len(weights))
    displacement = (co[2::3] - rest[2::3])[near]
    return float(displacement.std()) if displacement.size else 0.0


def build_sea(
    scene: Any, scenario: Scenario, camera: PinholeCamera, haze: Any
) -> SeaGrid:
    """Create the curved, wavy sea surface in front of the camera.

    The wave field has two FFT layers: long swell and short wind chop (which
    also produces the whitecap foam). Blender's ocean amplitude hardly depends
    on wind speed, so each layer is measured at unit scale and rescaled so the
    combined significant wave height matches the WMO Beaufort value.

    Args:
        scene: Scene receiving the sea.
        scenario: Scenario providing the sea state.
        camera: The camera; only its field of view is covered by the mesh.
        haze: Haze node group from :func:`build_haze_group`.

    Returns:
        The sea grid, which can sample the wave heights.
    """
    sea = scenario.sea
    radii, bearings = _polar_grid(camera, sea_extent(scenario))
    step = float(bearings[1] - bearings[0])
    horizon_m = horizon_distance(camera.pose.height_m)
    swell_length = sea.peak_wavelength_m
    chop_wind = min(sea.wind_speed_ms, 4.0)
    chop_length = max(0.834 * chop_wind**2, 0.5)
    # Rougher seas keep their geometric waves visible further towards the
    # horizon, so the mesh-resolution fade does not read as an unrealistic
    # calm band while whitecaps churn in the foreground.
    reach = 1.0 + 0.18 * sea.beaufort
    swell_fade, fade_start, fade_end = _wave_fades(
        radii, swell_length, step, horizon_m, reach
    )
    chop_fade, _, _ = _wave_fades(radii, chop_length, step, horizon_m, reach)
    land_offset, land_coverage = _land_offsets(
        radii, bearings, scenario.land, horizon_m
    )
    # Land shouldn't have ocean waves riding over it, regardless of how far
    # the wave-resolution fade above would otherwise reach.
    if land_coverage is None:
        mesh_chop_fade, mesh_swell_fade = chop_fade, swell_fade
        land_attr = np.zeros((len(radii), len(bearings)))
    else:
        mesh_chop_fade = chop_fade[:, None] * (1.0 - land_coverage)
        mesh_swell_fade = swell_fade[:, None] * (1.0 - land_coverage)
        land_attr = land_coverage
    mesh = _grid_mesh(
        "Sea",
        radii,
        bearings,
        {"fade": mesh_chop_fade, "fade_swell": mesh_swell_fade, "land": land_attr},
        height_offset=land_offset,
    )
    obj = bpy.data.objects.new("Sea", mesh)
    scene.collection.objects.link(obj)

    chop = _add_ocean(
        obj,
        "Chop",
        sea,
        chop_wind,
        np.clip(4.0 * chop_length, 10.0, 50.0),
        sea.choppiness,
        0,
    )
    if sea.foam_coverage > 0.0:
        chop.use_foam = True
        chop.foam_coverage = sea.foam_coverage
        chop.foam_layer_name = "foam"
    obj.modifiers.new("ChopFade", "NODES").node_group = _fade_node_group(
        "rest", "fade", store="rest_swell"
    )
    _add_ocean(
        obj,
        "Swell",
        sea,
        sea.wind_speed_ms,
        np.clip(6.0 * swell_length, 40.0, 600.0),
        0.5 * sea.choppiness,
        7919,
    )
    obj.modifiers.new("SwellFade", "NODES").node_group = _fade_node_group(
        "rest_swell", "fade_swell"
    )

    height = sea.significant_wave_height_m
    targets = {"Chop": 0.35 * height, "Swell": math.sqrt(1.0 - 0.35**2) * height}
    weights = {"Chop": chop_fade, "Swell": swell_fade}
    for layer, target in targets.items():
        spread = _measure_wave_spread(obj, layer, weights[layer])
        scale = target / (4.0 * spread) if spread > 1e-6 else 0.0
        obj.modifiers[layer].wave_scale = scale
    mesh.materials.append(_sea_material(scenario, haze))
    return SeaGrid(obj, radii, bearings, fade_start, fade_end)


def build_camera(scene: Any, camera: PinholeCamera, clip_end_m: float) -> Any:
    """Create the render camera matching a :class:`PinholeCamera`.

    Args:
        scene: Scene receiving the camera.
        camera: Camera model shared with the ground-truth computation.
        clip_end_m: Far clipping distance.

    Returns:
        The camera ``bpy.types.Object``, set as the scene camera.
    """
    data = bpy.data.cameras.new("Camera")
    data.sensor_fit = "HORIZONTAL"
    data.angle = math.radians(camera.hfov_deg)
    data.shift_x = data.shift_y = 0.0
    data.clip_start = 0.05
    data.clip_end = clip_end_m
    obj = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(obj)
    obj.matrix_world = Matrix(camera.pose.matrix_world())
    scene.camera = obj
    return obj
