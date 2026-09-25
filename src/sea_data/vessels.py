"""Procedural maritime objects and their placement on the sea surface.

Each builder produces one mesh with the bow along local +y, starboard along
local +x and the design waterline at ``z = 0``. Shapes, proportions and
colours are randomised from the object's ``variant_seed``.

This module must run inside Blender (it imports ``bpy``).
"""

import itertools
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from sea_data.geometry import EARTH_RADIUS_M, polar_to_xy
from sea_data.scenario import PlacedObject
from sea_data.scene_builder import SeaGrid, new_material

type RGB = tuple[float, float, float]
type Vec3 = tuple[float, float, float]

SHIP_HULL_COLORS: tuple[RGB, ...] = (
    (0.12, 0.012, 0.01),
    (0.012, 0.012, 0.014),
    (0.01, 0.025, 0.08),
    (0.015, 0.06, 0.03),
    (0.06, 0.08, 0.1),
    (0.35, 0.08, 0.01),
)
FISHING_HULL_COLORS: tuple[RGB, ...] = (
    (0.02, 0.08, 0.3),
    (0.4, 0.02, 0.015),
    (0.02, 0.2, 0.06),
    (0.7, 0.7, 0.68),
    (0.5, 0.18, 0.01),
    (0.015, 0.015, 0.02),
)
LEISURE_HULL_COLORS: tuple[RGB, ...] = (
    (0.8, 0.8, 0.78),
    (0.8, 0.8, 0.78),
    (0.8, 0.8, 0.78),
    (0.01, 0.03, 0.12),
    (0.3, 0.02, 0.02),
    (0.03, 0.04, 0.05),
)
ANTIFOULING_COLORS: tuple[RGB, ...] = (
    (0.2, 0.015, 0.01),
    (0.012, 0.012, 0.012),
    (0.02, 0.04, 0.12),
)
DECK_COLORS: tuple[RGB, ...] = (
    (0.15, 0.03, 0.02),
    (0.03, 0.08, 0.04),
    (0.12, 0.12, 0.12),
)
SUPERSTRUCTURE_COLORS: tuple[RGB, ...] = (
    (0.78, 0.78, 0.75),
    (0.7, 0.66, 0.55),
    (0.8, 0.8, 0.8),
)
CONTAINER_COLORS: tuple[RGB, ...] = (
    (0.01, 0.1, 0.3),
    (0.35, 0.02, 0.01),
    (0.02, 0.18, 0.05),
    (0.6, 0.18, 0.01),
    (0.25, 0.25, 0.25),
    (0.7, 0.7, 0.68),
    (0.6, 0.42, 0.02),
    (0.12, 0.05, 0.02),
    (0.02, 0.2, 0.3),
    (0.45, 0.02, 0.1),
)
WINDOW_COLOR: RGB = (0.01, 0.012, 0.015)
METAL_COLOR: RGB = (0.35, 0.35, 0.36)

U_PROFILE = (
    (0.0, -1.0),
    (0.75, -1.0),
    (0.97, -0.8),
    (1.0, -0.35),
    (1.0, 0.0),
    (1.02, 1.0),
)
ROUND_PROFILE = (
    (0.0, -1.0),
    (0.35, -0.92),
    (0.75, -0.6),
    (0.96, -0.2),
    (1.0, 0.0),
    (1.0, 1.0),
)
V_PROFILE = ((0.0, -1.0), (0.45, -0.6), (0.85, -0.2), (0.97, 0.0), (1.05, 1.0))


@dataclass(frozen=True)
class HullShape:
    """Parameters of a lofted hull.

    Attributes:
        length: Overall length.
        beam: Maximum width.
        draft: Depth of the keel below the waterline.
        freeboard: Height of the deck edge above the waterline amidships.
        profile: Half cross-section from the keel (x=0, z=-1) to the deck edge
            (z=+1); x is relative to the half beam, z to draft or freeboard.
        transom: Width of the stern relative to the beam.
        bow_start: Fraction of the length (from the stern) where the bow
            starts to taper.
        bow_power: Exponent of the bow taper; below 1 gives a blunt bow.
        bow_rise: Additional sheer at the bow relative to the freeboard.
        stations: Number of cross-sections along the hull.
    """

    length: float
    beam: float
    draft: float
    freeboard: float
    profile: tuple[tuple[float, float], ...]
    transom: float = 0.75
    bow_start: float = 0.55
    bow_power: float = 1.0
    bow_rise: float = 0.3
    stations: int = 28


def _ramp(t: float, start: float, end: float) -> float:
    return min(max((t - start) / (end - start), 0.0), 1.0)


def _beam_factor(t: float, shape: HullShape) -> float:
    if t < 0.18:
        return shape.transom + (1.0 - shape.transom) * math.sin(
            0.5 * math.pi * t / 0.18
        )
    if t <= shape.bow_start:
        return 1.0
    u = (t - shape.bow_start) / (1.0 - shape.bow_start)
    return max(math.cos(0.5 * math.pi * u), 0.0) ** shape.bow_power


def _rotate_z(point: Vec3, pivot: Vec3, angle: float) -> Vec3:
    dx, dy = point[0] - pivot[0], point[1] - pivot[1]
    c, s = math.cos(angle), math.sin(angle)
    return pivot[0] + c * dx - s * dy, pivot[1] + s * dx + c * dy, point[2]


@dataclass
class MeshBuilder:
    """Accumulates primitives with material slots into one bmesh.

    Attributes:
        bm: The bmesh being built.
        materials: Material slot definitions ``(name, colour, roughness, metal)``.
    """

    bm: Any = field(default_factory=bmesh.new)
    materials: list[tuple[str, RGB, float, float]] = field(default_factory=list)

    def material(
        self, name: str, color: RGB, roughness: float = 0.5, metallic: float = 0.0
    ) -> int:
        """Register a material slot.

        Args:
            name: Material name.
            color: Linear-RGB base colour.
            roughness: Surface roughness.
            metallic: Metalness.

        Returns:
            The slot index to pass to the primitive methods.
        """
        self.materials.append((name, color, roughness, metallic))
        return len(self.materials) - 1

    def _assign(self, verts: Sequence[Any], material: int) -> None:
        for face in {f for v in verts for f in v.link_faces}:
            face.material_index = material

    def box(
        self, center: Vec3, size: Vec3, material: int, rotation: Vec3 = (0.0, 0.0, 0.0)
    ) -> None:
        """Add a box.

        Args:
            center: Box centre.
            size: Extent along local x, y and z.
            material: Material slot.
            rotation: XYZ Euler rotation in radians.
        """
        matrix = Matrix.LocRotScale(Vector(center), Euler(rotation), Vector(size))
        geom = bmesh.ops.create_cube(self.bm, size=1.0, matrix=matrix)
        self._assign(geom["verts"], material)

    def cylinder(
        self,
        center: Vec3,
        radius: float,
        depth: float,
        material: int,
        radius_top: float | None = None,
        rotation: Vec3 = (0.0, 0.0, 0.0),
        segments: int = 16,
    ) -> None:
        """Add a (truncated) cone or cylinder along local z.

        Args:
            center: Centre of the axis.
            radius: Bottom radius.
            depth: Length along the axis.
            material: Material slot.
            radius_top: Top radius; defaults to ``radius``.
            rotation: XYZ Euler rotation in radians.
            segments: Number of sides.
        """
        matrix = Matrix.LocRotScale(Vector(center), Euler(rotation), None)
        geom = bmesh.ops.create_cone(
            self.bm,
            cap_ends=True,
            cap_tris=False,
            segments=segments,
            radius1=radius,
            radius2=radius if radius_top is None else radius_top,
            depth=depth,
            matrix=matrix,
        )
        self._assign(geom["verts"], material)

    def sphere(self, center: Vec3, radius: float, material: int) -> None:
        """Add a UV sphere.

        Args:
            center: Sphere centre.
            radius: Sphere radius.
            material: Material slot.
        """
        geom = bmesh.ops.create_uvsphere(
            self.bm,
            u_segments=16,
            v_segments=8,
            radius=radius,
            matrix=Matrix.Translation(Vector(center)),
        )
        self._assign(geom["verts"], material)

    def polygon(self, points: Sequence[Vec3], material: int) -> None:
        """Add a single flat polygon (e.g. a sail).

        Args:
            points: Polygon corners.
            material: Material slot.
        """
        face = self.bm.faces.new([self.bm.verts.new(p) for p in points])
        face.material_index = material

    def hull(self, shape: HullShape, topside: int, bottom: int, deck: int) -> None:
        """Add a closed hull lofted from cross-sections.

        Args:
            shape: Hull parameters.
            topside: Material slot above the waterline.
            bottom: Material slot below the waterline.
            deck: Material slot of the deck.
        """
        rings = []
        for k in range(shape.stations + 1):
            t = k / shape.stations
            y = (t - 0.5) * shape.length
            half = 0.5 * shape.beam * max(_beam_factor(t, shape), 0.02)
            keel = shape.draft * (1.0 - 0.6 * _ramp(t, 0.8, 1.0))
            deck_z = shape.freeboard * (1.0 + shape.bow_rise * _ramp(t, 0.6, 1.0) ** 2)
            starboard = [
                (xf * half, y, zf * (keel if zf < 0 else deck_z))
                for xf, zf in shape.profile
            ]
            port = [(-x, y_, z) for x, y_, z in reversed(starboard[1:])]
            rings.append([self.bm.verts.new(p) for p in starboard + port])
        count = len(rings[0])
        deck_edge = len(shape.profile) - 1
        for aft, fwd in itertools.pairwise(rings):
            for i in range(count):
                j = (i + 1) % count
                face = self.bm.faces.new((aft[i], aft[j], fwd[j], fwd[i]))
                if i == deck_edge:
                    face.material_index = deck
                else:
                    mid_z = 0.5 * (aft[i].co.z + aft[j].co.z)
                    face.material_index = bottom if mid_z < -1e-4 else topside
        self.bm.faces.new(list(reversed(rings[0]))).material_index = topside
        self.bm.faces.new(rings[-1]).material_index = topside

    def to_object(self, name: str, haze: Any) -> Any:
        """Convert the accumulated geometry into a Blender object.

        Args:
            name: Object and mesh name.
            haze: Haze node group for the materials.

        Returns:
            The new ``bpy.types.Object`` (not yet linked to a scene).
        """
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)
        mesh = bpy.data.meshes.new(name)
        self.bm.to_mesh(mesh)
        self.bm.free()
        for mat_name, color, roughness, metallic in self.materials:
            mesh.materials.append(
                new_material(f"{name}.{mat_name}", haze, color, roughness, metallic)
            )
        return bpy.data.objects.new(name, mesh)


@dataclass
class BuiltVessel:
    """Geometry of one object before placement.

    Attributes:
        mesh: The accumulated geometry.
        beam: Width used to sample the sea surface for rolling.
        heel_deg: Static heel (e.g. of a sailing yacht), positive to starboard.
    """

    mesh: MeshBuilder
    beam: float
    heel_deg: float = 0.0


def _superstructure_block(
    m: MeshBuilder,
    center: Vec3,
    size: Vec3,
    body: int,
    window: int,
    window_band: float = 0.3,
) -> None:
    m.box(center, size, body)
    band = (size[0] * 1.02, size[1] * 1.02, size[2] * window_band)
    m.box((center[0], center[1], center[2] + 0.15 * size[2]), band, window)


def build_cargo_ship(rng: random.Random, length: float) -> BuiltVessel:
    """Build a container ship, tanker or bulk carrier.

    Args:
        rng: Random source for proportions and colours.
        length: Overall length in metres.

    Returns:
        The unplaced vessel.
    """
    beam = length / rng.uniform(6.2, 7.6)
    freeboard = length / rng.uniform(24.0, 32.0)
    m = MeshBuilder()
    topside = m.material("hull", rng.choice(SHIP_HULL_COLORS), 0.45)
    bottom = m.material("bottom", rng.choice(ANTIFOULING_COLORS), 0.6)
    deck = m.material("deck", rng.choice(DECK_COLORS), 0.7)
    white = m.material("superstructure", rng.choice(SUPERSTRUCTURE_COLORS), 0.4)
    window = m.material("window", WINDOW_COLOR, 0.1)
    metal = m.material("metal", METAL_COLOR, 0.4, 0.6)
    funnel = m.material("funnel", rng.choice(CONTAINER_COLORS), 0.5)
    m.hull(
        HullShape(
            length,
            beam,
            length / rng.uniform(19.0, 25.0),
            freeboard,
            U_PROFILE,
            transom=0.8,
            bow_start=rng.uniform(0.62, 0.75),
            bow_power=0.8,
            bow_rise=0.2,
        ),
        topside,
        bottom,
        deck,
    )

    aft_fraction = rng.uniform(0.08, 0.16)
    acc_y = (aft_fraction - 0.5) * length + 0.04 * length
    acc_len = max(0.06 * length, 8.0)
    levels = int(np.clip(0.09 * length / 2.8, 3, 12))
    z = freeboard
    for level in range(levels):
        width = beam * (0.86 - 0.015 * level)
        _superstructure_block(
            m,
            (0.0, acc_y, z + 1.4),
            (width, acc_len * (1.0 - 0.02 * level), 2.8),
            white,
            window,
        )
        z += 2.8
    _superstructure_block(
        m,
        (0.0, acc_y + 0.1 * acc_len, z + 1.5),
        (beam, 0.4 * acc_len, 3.0),
        white,
        window,
        0.45,
    )
    funnel_h = rng.uniform(6.0, 12.0)
    m.box(
        (0.0, acc_y - 0.55 * acc_len, z + funnel_h / 2 - 2.0),
        (0.2 * beam, 0.35 * acc_len, funnel_h),
        funnel,
    )
    m.cylinder((0.0, acc_y, z + 7.0), 0.35, 8.0, metal)

    cargo_start = acc_y + 0.5 * acc_len + 3.0
    cargo_end = (0.5 - 0.1) * length
    variant = rng.random()
    if variant < 0.55:
        colors = rng.sample(CONTAINER_COLORS, k=6)
        slots = [m.material(f"container{i}", c, 0.55) for i, c in enumerate(colors)]
        rows = max(1, int(0.9 * beam / 2.5))
        bays = max(1, int((cargo_end - cargo_start) / 13.0))
        max_tiers = int(np.clip(levels - 1, 2, 9))
        for bay in range(bays):
            y = cargo_start + (bay + 0.5) * 13.0
            for row in range(rows):
                x = (row - (rows - 1) / 2.0) * 2.5
                for tier in range(rng.randint(max(0, max_tiers - 3), max_tiers)):
                    m.box(
                        (x, y, freeboard + 1.8 + tier * 2.6),
                        (2.44, 12.2, 2.59),
                        rng.choice(slots),
                    )
    elif variant < 0.8:
        m.box(
            (0.0, 0.5 * (cargo_start + cargo_end), freeboard + 1.2),
            (1.5, cargo_end - cargo_start, 1.2),
            metal,
        )
        m.box(
            (0.0, 0.5 * (cargo_start + cargo_end), freeboard + 1.5),
            (0.8 * beam, 3.0, 2.5),
            metal,
        )
    else:
        hatch_color = m.material("hatch", rng.choice(DECK_COLORS), 0.6)
        hatches = max(2, int((cargo_end - cargo_start) / (0.09 * length)))
        pitch = (cargo_end - cargo_start) / hatches
        for i in range(hatches):
            y = cargo_start + (i + 0.5) * pitch
            m.box(
                (0.0, y, freeboard + 1.0), (0.7 * beam, 0.75 * pitch, 2.0), hatch_color
            )
            if i % 2 == 0:
                post_h = 0.06 * length
                m.cylinder(
                    (0.0, y + 0.5 * pitch, freeboard + post_h / 2), 1.0, post_h, funnel
                )
                m.box(
                    (0.0, y + 0.5 * pitch + 0.2 * length / 2, freeboard + post_h),
                    (0.8, 0.2 * length, 0.8),
                    funnel,
                    (0.5, 0.0, 0.0),
                )
    m.cylinder((0.0, 0.44 * length, freeboard * 1.25 + 4.0), 0.3, 8.0, metal)
    return BuiltVessel(m, beam)


def build_fishing_boat(rng: random.Random, length: float) -> BuiltVessel:
    """Build a trawler-style fishing boat.

    Args:
        rng: Random source for proportions and colours.
        length: Overall length in metres.

    Returns:
        The unplaced vessel.
    """
    beam = length / rng.uniform(3.2, 4.0)
    freeboard = length / rng.uniform(9.0, 12.0)
    m = MeshBuilder()
    topside = m.material("hull", rng.choice(FISHING_HULL_COLORS), 0.45)
    bottom = m.material("bottom", rng.choice(ANTIFOULING_COLORS), 0.6)
    deck = m.material("deck", rng.choice(DECK_COLORS), 0.7)
    white = m.material("wheelhouse", rng.choice(SUPERSTRUCTURE_COLORS), 0.4)
    window = m.material("window", WINDOW_COLOR, 0.1)
    gear = m.material(
        "gear", rng.choice(((0.6, 0.2, 0.01), (0.6, 0.45, 0.02), METAL_COLOR)), 0.5
    )
    m.hull(
        HullShape(
            length,
            beam,
            length / rng.uniform(10.0, 13.0),
            freeboard,
            ROUND_PROFILE,
            transom=0.85,
            bow_power=0.7,
            bow_rise=0.6,
        ),
        topside,
        bottom,
        deck,
    )
    house_y = rng.uniform(0.05, 0.2) * length
    house_h = max(2.2, 0.12 * length)
    _superstructure_block(
        m,
        (0.0, house_y, freeboard * 1.05 + house_h / 2),
        (0.62 * beam, 0.2 * length, house_h),
        white,
        window,
        0.35,
    )
    top = freeboard * 1.05 + house_h
    m.box((0.0, house_y, top + 0.1), (0.66 * beam, 0.22 * length, 0.2), white)
    mast_h = 0.3 * length
    m.cylinder((0.0, house_y, top + mast_h / 2), 0.012 * length + 0.05, mast_h, gear)
    gantry_h = 0.25 * length
    for side in (-1.0, 1.0):
        m.box(
            (side * 0.38 * beam, -0.44 * length, freeboard + gantry_h / 2),
            (0.25, 0.25, gantry_h),
            gear,
        )
    m.box((0.0, -0.44 * length, freeboard + gantry_h), (0.8 * beam, 0.3, 0.3), gear)
    m.cylinder(
        (0.0, -0.32 * length, freeboard + 0.06 * length),
        0.06 * length,
        0.5 * beam,
        gear,
        rotation=(0.0, 0.5 * math.pi, 0.0),
    )
    if rng.random() < 0.5:
        for side in (-1.0, 1.0):
            m.cylinder(
                (side * 0.18 * length, house_y, top + 0.18 * length),
                0.06,
                0.5 * length,
                gear,
                rotation=(0.0, side * 0.8, 0.0),
                segments=6,
            )
    return BuiltVessel(m, beam)


def build_sailboat(rng: random.Random, length: float) -> BuiltVessel:
    """Build a sailing yacht, under sail or with furled sails.

    Args:
        rng: Random source for proportions, colours and sail trim.
        length: Overall length in metres.

    Returns:
        The unplaced vessel, including its heel angle when sailing.
    """
    beam = length / rng.uniform(3.0, 3.6)
    freeboard = length / rng.uniform(14.0, 18.0)
    draft = length / 28.0
    m = MeshBuilder()
    topside = m.material("hull", rng.choice(LEISURE_HULL_COLORS), 0.25)
    bottom = m.material("bottom", rng.choice(ANTIFOULING_COLORS), 0.6)
    deck = m.material("deck", rng.choice(((0.35, 0.2, 0.08), (0.6, 0.6, 0.58))), 0.6)
    white = m.material("cabin", (0.8, 0.8, 0.78), 0.3)
    window = m.material("window", WINDOW_COLOR, 0.1)
    metal = m.material("rig", METAL_COLOR, 0.3, 0.8)
    sailcloth = m.material(
        "sail",
        rng.choice(((0.8, 0.8, 0.78), (0.75, 0.7, 0.58), (0.65, 0.66, 0.68))),
        0.8,
    )
    m.hull(
        HullShape(
            length,
            beam,
            draft,
            freeboard,
            ROUND_PROFILE,
            transom=0.65,
            bow_start=0.45,
            bow_power=1.3,
            bow_rise=0.15,
        ),
        topside,
        bottom,
        deck,
    )
    m.box(
        (0.0, -0.02 * length, -draft - 0.06 * length),
        (max(0.08, 0.012 * length), 0.22 * length, 0.12 * length),
        bottom,
    )
    m.box(
        (0.0, -0.42 * length, -draft - 0.02 * length),
        (0.05, 0.06 * length, 0.08 * length),
        bottom,
    )
    _superstructure_block(
        m,
        (0.0, 0.0, freeboard + 0.035 * length),
        (0.55 * beam, 0.34 * length, 0.07 * length),
        white,
        window,
        0.3,
    )

    mast_y = 0.1 * length
    mast_h = 1.3 * length
    boom_z = freeboard + 0.12 * length
    m.cylinder(
        (0.0, mast_y, freeboard + mast_h / 2),
        0.008 * length + 0.03,
        mast_h,
        metal,
        segments=8,
    )
    heel = 0.0
    if rng.random() < 0.7:
        trim = math.radians(rng.uniform(8.0, 35.0)) * rng.choice((-1.0, 1.0))
        pivot = (0.0, mast_y, 0.0)
        clew = _rotate_z((0.0, mast_y - 0.37 * length, boom_z), pivot, trim)
        m.polygon(
            (
                (0.0, mast_y - 0.01, boom_z),
                (0.0, mast_y - 0.01, freeboard + 0.97 * mast_h),
                clew,
            ),
            sailcloth,
        )
        jib_clew = _rotate_z(
            (0.0, mast_y - 0.1 * length, freeboard + 0.1 * length),
            (0.0, 0.47 * length, 0.0),
            0.8 * trim,
        )
        m.polygon(
            (
                (0.0, mast_y + 0.02, freeboard + 0.8 * mast_h),
                (0.0, 0.47 * length, freeboard + 0.05 * length),
                jib_clew,
            ),
            sailcloth,
        )
        boom_mid = _rotate_z((0.0, mast_y - 0.19 * length, boom_z), pivot, trim)
        m.box(
            boom_mid,
            (0.03 * length, 0.38 * length, 0.025 * length),
            metal,
            (0.0, 0.0, trim),
        )
        heel = math.copysign(rng.uniform(4.0, 20.0), trim)
    else:
        cover = m.material(
            "sail_cover", rng.choice(((0.02, 0.05, 0.2), (0.3, 0.3, 0.3))), 0.8
        )
        m.box(
            (0.0, mast_y - 0.19 * length, boom_z),
            (0.03 * length, 0.38 * length, 0.025 * length),
            metal,
        )
        m.cylinder(
            (0.0, mast_y - 0.19 * length, boom_z + 0.03 * length),
            0.03 * length,
            0.36 * length,
            cover,
            rotation=(0.5 * math.pi, 0.0, 0.0),
        )
    return BuiltVessel(m, beam, heel)


def build_motorboat(rng: random.Random, length: float) -> BuiltVessel:
    """Build a planing motorboat: cabin cruiser or centre console.

    Args:
        rng: Random source for proportions and colours.
        length: Overall length in metres.

    Returns:
        The unplaced vessel.
    """
    beam = length / rng.uniform(2.7, 3.2)
    freeboard = length / rng.uniform(8.0, 10.0)
    m = MeshBuilder()
    topside = m.material("hull", rng.choice(LEISURE_HULL_COLORS), 0.2)
    bottom = m.material(
        "bottom", rng.choice(ANTIFOULING_COLORS + ((0.8, 0.8, 0.78),)), 0.4
    )
    deck = m.material("deck", (0.75, 0.75, 0.72), 0.4)
    white = m.material("cabin", (0.8, 0.8, 0.78), 0.25)
    window = m.material("window", WINDOW_COLOR, 0.05)
    engine = m.material(
        "engine", rng.choice(((0.02, 0.02, 0.02), (0.7, 0.7, 0.7))), 0.3
    )
    m.hull(
        HullShape(
            length,
            beam,
            length / rng.uniform(16.0, 20.0),
            freeboard,
            V_PROFILE,
            transom=0.95,
            bow_start=0.5,
            bow_power=1.1,
            bow_rise=0.3,
        ),
        topside,
        bottom,
        deck,
    )
    if rng.random() < 0.55:
        _superstructure_block(
            m,
            (0.0, 0.05 * length, freeboard + 0.055 * length),
            (0.72 * beam, 0.38 * length, 0.11 * length),
            white,
            window,
            0.45,
        )
    else:
        m.box(
            (0.0, 0.0, freeboard + 0.05 * length),
            (0.3 * beam, 0.12 * length, 0.1 * length),
            white,
        )
        roof_z = freeboard + 0.2 * length
        m.box(
            (0.0, -0.02 * length, roof_z),
            (0.5 * beam, 0.2 * length, 0.02 * length),
            white,
        )
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                m.cylinder(
                    (
                        sx * 0.22 * beam,
                        -0.02 * length + sy * 0.09 * length,
                        freeboard + 0.1 * length,
                    ),
                    0.03,
                    0.2 * length,
                    engine,
                    segments=6,
                )
    m.box(
        (0.0, -0.53 * length, freeboard * 0.4),
        (0.12 * beam, 0.08 * length, 0.14 * length),
        engine,
    )
    return BuiltVessel(m, beam)


BUOY_STYLES: tuple[tuple[str, tuple[RGB, ...], str], ...] = (
    ("lateral_port", ((0.55, 0.02, 0.015),), "can"),
    ("lateral_starboard", ((0.015, 0.25, 0.04),), "cone"),
    ("cardinal", ((0.7, 0.5, 0.02), (0.015, 0.015, 0.015)), "double_cone"),
    ("special", ((0.7, 0.5, 0.02),), "cross"),
    ("safe_water", ((0.55, 0.02, 0.015), (0.8, 0.8, 0.8)), "sphere"),
)


def build_buoy(rng: random.Random, length: float) -> BuiltVessel:
    """Build a navigation buoy with an IALA-style top mark.

    Args:
        rng: Random source for the buoy style.
        length: Total height above the waterline in metres.

    Returns:
        The unplaced buoy.
    """
    _name, colors, topmark = rng.choice(BUOY_STYLES)
    h = length
    m = MeshBuilder()
    slots = [m.material(f"paint{i}", c, 0.5) for i, c in enumerate(colors)]
    dark = m.material("topmark", colors[-1] if topmark != "sphere" else colors[0], 0.5)
    lamp = m.material("lamp", (0.8, 0.75, 0.6), 0.2)
    m.cylinder((0.0, 0.0, 0.04 * h), 0.26 * h, 0.32 * h, slots[0], segments=20)
    bands = len(slots)
    for i in range(bands):
        z0 = 0.2 * h + i * 0.58 * h / bands
        z1 = 0.2 * h + (i + 1) * 0.58 * h / bands
        r0 = 0.17 * h - 0.1 * h * (z0 - 0.2 * h) / (0.58 * h)
        r1 = 0.17 * h - 0.1 * h * (z1 - 0.2 * h) / (0.58 * h)
        m.cylinder(
            (0.0, 0.0, 0.5 * (z0 + z1)),
            r0,
            z1 - z0,
            slots[i],
            radius_top=r1,
            segments=16,
        )
    top = 0.78 * h
    if topmark == "can":
        m.cylinder((0.0, 0.0, top + 0.08 * h), 0.07 * h, 0.12 * h, dark)
    elif topmark == "cone":
        m.cylinder((0.0, 0.0, top + 0.08 * h), 0.09 * h, 0.14 * h, dark, radius_top=0.0)
    elif topmark == "double_cone":
        m.cylinder((0.0, 0.0, top + 0.04 * h), 0.07 * h, 0.08 * h, dark, radius_top=0.0)
        m.cylinder((0.0, 0.0, top + 0.14 * h), 0.07 * h, 0.08 * h, dark, radius_top=0.0)
    elif topmark == "cross":
        m.box(
            (0.0, 0.0, top + 0.1 * h),
            (0.16 * h, 0.03 * h, 0.03 * h),
            dark,
            (0.0, 0.8, 0.0),
        )
        m.box(
            (0.0, 0.0, top + 0.1 * h),
            (0.16 * h, 0.03 * h, 0.03 * h),
            dark,
            (0.0, -0.8, 0.0),
        )
    else:
        m.sphere((0.0, 0.0, top + 0.1 * h), 0.07 * h, dark)
    m.cylinder((0.0, 0.0, 0.99 * h), 0.03 * h, 0.04 * h, lamp, segments=8)
    return BuiltVessel(m, 0.52 * h)


BUILDERS: dict[str, Callable[[random.Random, float], BuiltVessel]] = {
    "cargo_ship": build_cargo_ship,
    "fishing_boat": build_fishing_boat,
    "sailboat": build_sailboat,
    "motorboat": build_motorboat,
    "buoy": build_buoy,
}
"""Object builders keyed by class name."""


@dataclass
class SceneObject:
    """An object instantiated in the scene.

    Attributes:
        obj: The Blender object.
        placed: The scenario entry it was built from.
        instance_id: Instance id used in the masks (1-254).
    """

    obj: Any
    placed: PlacedObject
    instance_id: int


def _placement_matrix(placed: PlacedObject, z: float, pitch: float, roll: float) -> Any:
    x, y = polar_to_xy(placed.distance_m, placed.bearing_deg)
    bearing = math.radians(placed.bearing_deg)
    curvature = Matrix.Rotation(
        placed.distance_m / EARTH_RADIUS_M,
        4,
        Vector((-math.cos(bearing), math.sin(bearing), 0.0)),
    )
    heading = Matrix.Rotation(-math.radians(placed.heading_deg), 4, "Z")
    return (
        Matrix.Translation(Vector((x, y, z)))
        @ curvature
        @ heading
        @ Matrix.Rotation(pitch, 4, "X")
        @ Matrix.Rotation(roll, 4, "Y")
    )


def build_objects(
    scene: Any, objects: Sequence[PlacedObject], sea: SeaGrid, haze: Any
) -> list[SceneObject]:
    """Build every scenario object and float it on the wavy sea surface.

    Heave, pitch and roll follow the evaluated wave heights sampled at the
    object's bow, stern and both sides.

    Args:
        scene: Scene receiving the objects.
        objects: Objects to build.
        sea: Sea grid used to sample wave heights.
        haze: Haze node group for the materials.

    Returns:
        The instantiated objects with instance ids starting at 1.
    """
    built = [
        BUILDERS[p.category](random.Random(p.variant_seed), p.length_m) for p in objects
    ]
    samples: list[Any] = []
    for placed, vessel in zip(objects, built, strict=True):
        x, y = polar_to_xy(placed.distance_m, placed.bearing_deg)
        heading = math.radians(placed.heading_deg)
        fwd = np.array((math.sin(heading), math.cos(heading)))
        right = np.array((math.cos(heading), -math.sin(heading)))
        centre = np.array((x, y))
        half_len = 0.35 * placed.length_m
        half_beam = 0.5 * vessel.beam
        samples.extend(
            (
                centre,
                centre + half_len * fwd,
                centre - half_len * fwd,
                centre - half_beam * right,
                centre + half_beam * right,
            )
        )
    heights = sea.surface_heights(np.array(samples).reshape(-1, 2)) if samples else []

    result = []
    for i, (placed, vessel) in enumerate(zip(objects, built, strict=True)):
        z_c, z_bow, z_stern, z_port, z_stbd = heights[5 * i : 5 * i + 5]
        pitch = 0.8 * math.atan2(z_bow - z_stern, 0.7 * placed.length_m)
        roll = 0.8 * math.atan2(z_port - z_stbd, vessel.beam)
        pitch = float(np.clip(pitch, -0.25, 0.25))
        roll = float(np.clip(roll, -0.45, 0.45)) + math.radians(vessel.heel_deg)
        z = 0.5 * z_c + 0.125 * (z_bow + z_stern + z_port + z_stbd)
        obj = vessel.mesh.to_object(f"{placed.category}.{i + 1:03d}", haze)
        scene.collection.objects.link(obj)
        obj.matrix_world = _placement_matrix(placed, float(z), pitch, roll)
        result.append(SceneObject(obj, placed, i + 1))
    return result
