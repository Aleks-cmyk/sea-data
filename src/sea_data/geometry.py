"""Camera, Earth-curvature and horizon geometry.

All code in this module is pure Python so that it can be unit-tested without
Blender and shared verbatim between the scene builder and the ground-truth
exporter.

Conventions:
    * World frame: ``x`` points east, ``y`` points north, ``z`` points up. The
      camera sits at ``(0, 0, height)`` above the mean sea level.
    * Bearings/yaw are compass angles in degrees, clockwise from north (+y).
    * Pitch is positive when the camera looks up.
    * Roll is positive when the camera's right-hand side dips down
      (starboard-down for a forward-looking ship camera).
    * Camera frame follows Blender: ``x`` right, ``y`` up, looking along -z.
    * Image frame: origin in the top-left corner, ``u`` to the right, ``v``
      downwards, pixel ``(i, j)`` covers ``[i, i + 1) x [j, j + 1)``.
    * The sea surface is modelled as the paraboloid ``z = -r^2 / (2 R)``,
      which is the second-order approximation of a sphere of radius ``R``.
      With this model the horizon dip satisfies ``tan(dip) = sqrt(2 h / R)``.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

EARTH_RADIUS_M = 6_371_000.0
"""Mean Earth radius in metres."""

MIN_HORIZON_SEGMENT_PX = 8.0
"""Shortest in-image horizon segment still reported as visible.

A line that only grazes a corner of the image (e.g. because the camera pitch
pushes the horizon almost entirely above or below the frame) would otherwise
satisfy the "at least two sample points inside the image" rule while being a
sliver no human could actually point to."""

type Vec3 = tuple[float, float, float]
type Vec2 = tuple[float, float]
type Mat3 = tuple[Vec3, Vec3, Vec3]


def earth_drop(distance_m: float, radius_m: float = EARTH_RADIUS_M) -> float:
    """Return how far the sea surface drops below the tangent plane.

    Args:
        distance_m: Horizontal distance from the camera nadir in metres.
        radius_m: Earth radius in metres.

    Returns:
        The drop ``d^2 / (2 R)`` in metres.
    """
    return distance_m * distance_m / (2.0 * radius_m)


def horizon_distance(height_m: float, radius_m: float = EARTH_RADIUS_M) -> float:
    """Return the distance to the geometric horizon for an observer.

    Args:
        height_m: Eye height above mean sea level in metres.
        radius_m: Earth radius in metres.

    Returns:
        The horizontal distance ``sqrt(2 R h)`` to the tangent point in metres.
    """
    return math.sqrt(2.0 * radius_m * max(height_m, 0.0))


def horizon_dip(height_m: float, radius_m: float = EARTH_RADIUS_M) -> float:
    """Return the dip of the geometric horizon below the horizontal plane.

    Args:
        height_m: Eye height above mean sea level in metres.
        radius_m: Earth radius in metres.

    Returns:
        The dip angle in radians.
    """
    return math.atan(math.sqrt(2.0 * max(height_m, 0.0) / radius_m))


def polar_to_xy(distance_m: float, bearing_deg: float) -> Vec2:
    """Convert a range/bearing pair relative to the camera into world x/y.

    Args:
        distance_m: Horizontal range in metres.
        bearing_deg: Compass bearing in degrees.

    Returns:
        The ``(x, y)`` world coordinates in metres.
    """
    bearing = math.radians(bearing_deg)
    return distance_m * math.sin(bearing), distance_m * math.cos(bearing)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass(frozen=True)
class CameraPose:
    """Position and orientation of the camera.

    Attributes:
        height_m: Height of the optical centre above mean sea level.
        yaw_deg: Compass heading of the optical axis.
        pitch_deg: Elevation of the optical axis, positive upwards.
        roll_deg: Rotation about the optical axis, positive right-side-down.
    """

    height_m: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def axes(self) -> Mat3:
        """Return the camera axes expressed in world coordinates.

        Returns:
            The ``(right, up, back)`` unit vectors. ``back`` points opposite
            to the viewing direction, matching Blender's camera frame.
        """
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        roll = math.radians(self.roll_deg)
        sy, cy = math.sin(yaw), math.cos(yaw)
        sp, cp = math.sin(pitch), math.cos(pitch)
        sr, cr = math.sin(roll), math.cos(roll)
        forward = (sy * cp, cy * cp, sp)
        right0 = (cy, -sy, 0.0)
        up0 = (-sy * sp, -cy * sp, cp)
        right = tuple(cr * r - sr * u for r, u in zip(right0, up0, strict=True))
        up = tuple(sr * r + cr * u for r, u in zip(right0, up0, strict=True))
        back = (-forward[0], -forward[1], -forward[2])
        return (right[0], right[1], right[2]), (up[0], up[1], up[2]), back

    def matrix_world(self) -> tuple[tuple[float, float, float, float], ...]:
        """Return the 4x4 camera-to-world matrix in row-major order.

        Returns:
            A matrix suitable for ``mathutils.Matrix`` and Blender's
            ``Object.matrix_world``.
        """
        right, up, back = self.axes()
        return (
            (right[0], up[0], back[0], 0.0),
            (right[1], up[1], back[1], 0.0),
            (right[2], up[2], back[2], self.height_m),
            (0.0, 0.0, 0.0, 1.0),
        )


@dataclass(frozen=True)
class PinholeCamera:
    """An ideal pinhole camera with square pixels and a centred principal point.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        hfov_deg: Horizontal field of view in degrees.
        pose: Camera position and orientation.
    """

    width: int
    height: int
    hfov_deg: float
    pose: CameraPose

    @property
    def focal_px(self) -> float:
        """Focal length in pixels."""
        return 0.5 * self.width / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view in degrees."""
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.focal_px))

    @property
    def principal_point(self) -> Vec2:
        """Principal point ``(cx, cy)`` in pixels."""
        return self.width / 2.0, self.height / 2.0

    def intrinsics(self) -> dict[str, float]:
        """Return the intrinsic parameters as a serialisable mapping.

        Returns:
            A mapping with ``fx``, ``fy``, ``cx`` and ``cy`` in pixels.
        """
        cx, cy = self.principal_point
        return {"fx": self.focal_px, "fy": self.focal_px, "cx": cx, "cy": cy}

    def project_direction(self, direction: Vec3) -> Vec2 | None:
        """Project a world-space direction (a point at infinity) to pixels.

        Args:
            direction: Direction vector in world coordinates.

        Returns:
            The ``(u, v)`` pixel coordinates, or ``None`` if the direction
            points behind the camera.
        """
        right, up, back = self.pose.axes()
        depth = -_dot(direction, back)
        if depth <= 1e-9:
            return None
        cx, cy = self.principal_point
        f = self.focal_px
        return (
            cx + f * _dot(direction, right) / depth,
            cy - f * _dot(direction, up) / depth,
        )

    def project_point(self, point: Vec3) -> Vec2 | None:
        """Project a world-space point to pixels.

        Args:
            point: Point in world coordinates (metres).

        Returns:
            The ``(u, v)`` pixel coordinates, or ``None`` if the point lies
            behind the camera.
        """
        offset = (point[0], point[1], point[2] - self.pose.height_m)
        return self.project_direction(offset)

    def contains(self, pixel: Vec2, margin: float = 0.0) -> bool:
        """Check whether a pixel coordinate lies inside the image.

        Args:
            pixel: The ``(u, v)`` coordinate.
            margin: Extra tolerance in pixels around the image border.

        Returns:
            ``True`` if the coordinate is inside the (expanded) image.
        """
        u, v = pixel
        return (
            -margin <= u <= self.width + margin and -margin <= v <= self.height + margin
        )


@dataclass(frozen=True)
class HorizonLine:
    """Ground-truth horizon of one image.

    Attributes:
        visible: Whether the horizon crosses the image.
        dip_deg: Dip of the geometric horizon below the horizontal plane.
        distance_m: Distance to the geometric horizon.
        endpoints: The best-fit line clipped to the image borders.
        y_left: Line ordinate at ``u = 0`` (may be outside the image).
        y_right: Line ordinate at ``u = width`` (may be outside the image).
        angle_deg: Image-space angle of the line, in ``(-90, 90]``, with
            positive values rising towards the right.
        offset_px: Signed distance from the image centre to the line, positive
            when the line lies below the centre.
        points: Sampled points of the exact (slightly curved) horizon.
        max_fit_error_px: Largest distance between ``points`` and the line.
    """

    visible: bool
    dip_deg: float
    distance_m: float
    endpoints: tuple[Vec2, Vec2] | None = None
    y_left: float | None = None
    y_right: float | None = None
    angle_deg: float | None = None
    offset_px: float | None = None
    points: tuple[Vec2, ...] = ()
    max_fit_error_px: float | None = None


def fit_line(points: Sequence[Vec2]) -> tuple[Vec2, Vec2]:
    """Fit a 2D line with total least squares.

    Args:
        points: At least two distinct points.

    Returns:
        A point on the line (the centroid) and the unit direction vector.

    Raises:
        ValueError: If fewer than two points are given.
    """
    if len(points) < 2:
        raise ValueError("fit_line needs at least two points")
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return (mx, my), (math.cos(theta), math.sin(theta))


def clip_line_to_rect(
    origin: Vec2, direction: Vec2, width: float, height: float
) -> tuple[Vec2, Vec2] | None:
    """Clip an infinite line to the rectangle ``[0, width] x [0, height]``.

    Args:
        origin: Any point on the line.
        direction: Direction vector of the line.
        width: Rectangle width.
        height: Rectangle height.

    Returns:
        The two intersection points ordered along ``direction``, or ``None``
        if the line misses the rectangle.
    """
    t_min, t_max = -math.inf, math.inf
    for p, d, upper in (
        (origin[0], direction[0], width),
        (origin[1], direction[1], height),
    ):
        if abs(d) < 1e-12:
            if not 0.0 <= p <= upper:
                return None
            continue
        t0, t1 = sorted(((0.0 - p) / d, (upper - p) / d))
        t_min, t_max = max(t_min, t0), min(t_max, t1)
    if t_min > t_max:
        return None
    return (
        (origin[0] + t_min * direction[0], origin[1] + t_min * direction[1]),
        (origin[0] + t_max * direction[0], origin[1] + t_max * direction[1]),
    )


def _horizon_directions(dip: float, samples: int) -> Iterable[Vec3]:
    cos_dip, sin_dip = math.cos(dip), math.sin(dip)
    for i in range(samples):
        bearing = 2.0 * math.pi * i / samples
        yield (math.sin(bearing) * cos_dip, math.cos(bearing) * cos_dip, -sin_dip)


def horizon_line(
    camera: PinholeCamera,
    radius_m: float = EARTH_RADIUS_M,
    samples: int = 3600,
    max_points: int = 33,
) -> HorizonLine:
    """Compute the ground-truth horizon line for a camera.

    The true horizon is the cone of directions with constant dip below the
    horizontal. It is sampled densely, projected into the image and fitted
    with a straight line, which is how horizon datasets usually encode it.

    Args:
        camera: The camera observing the sea.
        radius_m: Earth radius in metres.
        samples: Number of bearings sampled around the full circle.
        max_points: Maximum number of curve points kept in the result.

    Returns:
        The horizon annotation. ``visible`` is ``False`` when fewer than two
        sampled horizon points fall inside the image, or when the horizon
        only grazes a corner for less than :data:`MIN_HORIZON_SEGMENT_PX`.
    """
    height_m = camera.pose.height_m
    dip = horizon_dip(height_m, radius_m)
    base = HorizonLine(
        visible=False,
        dip_deg=math.degrees(dip),
        distance_m=horizon_distance(height_m, radius_m),
    )
    projected = (camera.project_direction(d) for d in _horizon_directions(dip, samples))
    inside = [p for p in projected if p is not None and camera.contains(p)]
    if len(inside) < 2:
        return base
    inside.sort()
    origin, direction = fit_line(inside)
    if direction[0] < 0:
        direction = (-direction[0], -direction[1])
    endpoints = clip_line_to_rect(origin, direction, camera.width, camera.height)
    if endpoints is None:
        return base
    (ex0, ey0), (ex1, ey1) = endpoints
    if math.hypot(ex1 - ex0, ey1 - ey0) < MIN_HORIZON_SEGMENT_PX:
        return base

    normal = (-direction[1], direction[0])
    cx, cy = camera.principal_point
    offset = normal[0] * (origin[0] - cx) + normal[1] * (origin[1] - cy)
    fit_error = max(
        abs(normal[0] * (p[0] - origin[0]) + normal[1] * (p[1] - origin[1]))
        for p in inside
    )
    y_left = y_right = None
    if abs(direction[0]) > 1e-9:
        slope = direction[1] / direction[0]
        y_left = origin[1] + slope * (0.0 - origin[0])
        y_right = origin[1] + slope * (camera.width - origin[0])
    angle = -math.degrees(math.atan2(direction[1], direction[0]))
    if angle <= -90.0:
        angle += 180.0
    step = max(1, math.ceil(len(inside) / max_points))
    kept = inside[::step]
    if kept[-1] != inside[-1]:
        kept.append(inside[-1])
    return HorizonLine(
        visible=True,
        dip_deg=base.dip_deg,
        distance_m=base.distance_m,
        endpoints=endpoints,
        y_left=y_left,
        y_right=y_right,
        angle_deg=angle,
        offset_px=offset,
        points=tuple(kept),
        max_fit_error_px=fit_error,
    )
