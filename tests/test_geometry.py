import math

import pytest

from sea_data.geometry import (
    CameraPose,
    PinholeCamera,
    clip_line_to_rect,
    earth_drop,
    fit_line,
    horizon_dip,
    horizon_distance,
    horizon_line,
    polar_to_xy,
)


def _cross(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def test_earth_curvature_values() -> None:
    assert horizon_distance(10.0) == pytest.approx(11_288, rel=1e-3)
    assert math.degrees(horizon_dip(10.0)) == pytest.approx(0.1016, rel=1e-2)
    assert earth_drop(horizon_distance(10.0)) == pytest.approx(10.0)
    assert horizon_distance(-1.0) == 0.0


def test_polar_to_xy_uses_compass_bearings() -> None:
    assert polar_to_xy(100.0, 0.0) == pytest.approx((0.0, 100.0))
    assert polar_to_xy(100.0, 90.0) == pytest.approx((100.0, 0.0))


@pytest.mark.parametrize(
    "pose",
    [
        CameraPose(5.0),
        CameraPose(5.0, 37.0, -12.0, 8.0),
        CameraPose(20.0, 250.0, 30.0, -45.0),
    ],
)
def test_camera_axes_are_right_handed_orthonormal(pose: CameraPose) -> None:
    right, up, back = pose.axes()
    for axis in (right, up, back):
        assert sum(c * c for c in axis) == pytest.approx(1.0)
    assert sum(r * u for r, u in zip(right, up, strict=True)) == pytest.approx(0.0)
    assert _cross(right, up) == pytest.approx(back)


def test_matrix_world_places_camera_at_height() -> None:
    matrix = CameraPose(12.5, 30.0, 5.0, 2.0).matrix_world()
    assert [row[3] for row in matrix] == pytest.approx([0.0, 0.0, 12.5, 1.0])


def test_projection_of_level_camera() -> None:
    camera = PinholeCamera(200, 100, 90.0, CameraPose(0.0, yaw_deg=90.0))
    assert camera.focal_px == pytest.approx(100.0)
    assert camera.project_direction((1.0, 0.0, 0.0)) == pytest.approx((100.0, 50.0))
    assert camera.project_direction((1.0, -1.0, 0.0)) == pytest.approx((200.0, 50.0))
    assert camera.project_direction((1.0, 0.0, 0.5)) == pytest.approx((100.0, 0.0))
    assert camera.project_direction((-1.0, 0.0, 0.0)) is None
    assert camera.project_point((10.0, 0.0, 0.0)) == pytest.approx((100.0, 50.0))


def test_vfov_and_intrinsics() -> None:
    camera = PinholeCamera(200, 100, 90.0, CameraPose(1.0))
    assert camera.vfov_deg == pytest.approx(2 * math.degrees(math.atan(0.5)))
    assert camera.intrinsics() == pytest.approx(
        {"fx": 100.0, "fy": 100.0, "cx": 100.0, "cy": 50.0}
    )


def test_level_horizon_sits_below_centre_by_dip() -> None:
    camera = PinholeCamera(1280, 720, 60.0, CameraPose(20.0))
    horizon = horizon_line(camera)
    assert horizon.visible
    assert horizon.angle_deg == pytest.approx(0.0, abs=1e-6)
    # The true horizon curves slightly downwards towards the image edges.
    centre = camera.focal_px * math.tan(horizon_dip(20.0))
    edge = centre / math.cos(math.radians(30.0))
    assert horizon.offset_px is not None
    assert centre <= horizon.offset_px <= edge
    assert horizon.y_left == pytest.approx(horizon.y_right)
    assert horizon.max_fit_error_px is not None
    assert horizon.max_fit_error_px < 1.0


def test_pitch_moves_horizon_down_and_roll_tilts_it() -> None:
    level_camera = PinholeCamera(640, 480, 60.0, CameraPose(5.0))
    level = horizon_line(level_camera)
    pitched = horizon_line(PinholeCamera(640, 480, 60.0, CameraPose(5.0, 0, 5.0)))
    rolled = horizon_line(PinholeCamera(640, 480, 60.0, CameraPose(5.0, 0, 0, 10.0)))
    assert level.offset_px is not None and pitched.offset_px is not None
    expected = level_camera.focal_px * math.tan(math.radians(5.0))
    assert pitched.offset_px - level.offset_px == pytest.approx(expected, rel=0.02)
    assert rolled.angle_deg == pytest.approx(10.0, abs=0.1)
    assert rolled.y_right is not None and rolled.y_left is not None
    assert rolled.y_right < rolled.y_left


def test_horizon_hidden_when_looking_at_the_sky() -> None:
    camera = PinholeCamera(640, 480, 40.0, CameraPose(5.0, pitch_deg=60.0))
    horizon = horizon_line(camera)
    assert not horizon.visible
    assert horizon.endpoints is None
    assert horizon.dip_deg > 0


def test_horizon_hidden_when_only_grazing_a_corner() -> None:
    camera = PinholeCamera(
        640, 360, 70.0, CameraPose(height_m=20.0, pitch_deg=33.5, roll_deg=27.5)
    )
    horizon = horizon_line(camera)
    assert not horizon.visible
    assert horizon.endpoints is None


def test_fit_line_and_clipping() -> None:
    origin, direction = fit_line([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
    assert origin == pytest.approx((1.0, 2.0))
    assert abs(direction[1] / direction[0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fit_line([(0.0, 0.0)])
    assert clip_line_to_rect((0.0, 5.0), (1.0, 0.0), 10.0, 10.0) == (
        (0.0, 5.0),
        (10.0, 5.0),
    )
    assert clip_line_to_rect((0.0, 20.0), (1.0, 0.0), 10.0, 10.0) is None
    assert clip_line_to_rect((0.0, 0.0), (1.0, 1.0), 10.0, 5.0) == (
        (0.0, 0.0),
        (5.0, 5.0),
    )
