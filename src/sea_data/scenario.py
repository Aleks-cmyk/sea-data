"""Random sampling of fully specified maritime scenarios.

A :class:`Scenario` holds every random choice needed to build one image, so it
can be serialised to JSON, handed to a Blender worker and reproduced exactly.
"""
from __future__ import annotations


import itertools
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sea_data.config import SimulatorConfig
from sea_data.geometry import CameraPose, horizon_distance

BEAUFORT_WIND_MS: tuple[tuple[float, float], ...] = (
    (0.0, 0.5),
    (0.5, 1.5),
    (1.6, 3.3),
    (3.4, 5.4),
    (5.5, 7.9),
    (8.0, 10.7),
    (10.8, 13.8),
    (13.9, 17.1),
    (17.2, 20.7),
)
"""Wind speed range in m/s for Beaufort numbers 0 to 8."""

BEAUFORT_WAVE_HEIGHT_M: tuple[float, ...] = (
    0.05,
    0.1,
    0.2,
    0.6,
    1.0,
    2.0,
    3.0,
    4.0,
    5.5,
)
"""Probable significant wave height in metres (WMO) for Beaufort 0 to 8.

Beaufort 0 keeps a small residual swell instead of a perfect mirror.
"""

OBJECT_HEIGHT_RATIO: dict[str, float] = {
    "cargo_ship": 0.18,
    "fishing_boat": 0.4,
    "sailboat": 1.35,
    "motorboat": 0.3,
    "buoy": 1.0,
}
"""Approximate height above the waterline relative to the object length."""

WATER_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.003, 0.018, 0.045),
    (0.008, 0.028, 0.035),
    (0.010, 0.040, 0.032),
    (0.018, 0.024, 0.028),
)
"""Linear-RGB base colours: deep ocean, North Atlantic, coastal, grey."""

_SEPARATION_ATTEMPTS = 25


@dataclass(frozen=True)
class Sun:
    """Sun position.

    Attributes:
        elevation_deg: Elevation above the horizontal plane.
        azimuth_deg: Compass bearing of the sun.
    """

    elevation_deg: float
    azimuth_deg: float


@dataclass(frozen=True)
class Atmosphere:
    """Sky and visibility conditions.

    Attributes:
        weather: Name of the weather preset.
        visibility_m: Meteorological visibility (2 % contrast threshold).
        cloud_cover: Fraction of the sky covered by clouds.
        aerosol_density: Aerosol density of the physical sky model.
        cloud_seed: Seed that offsets the procedural cloud pattern.
    """

    weather: str
    visibility_m: float
    cloud_cover: float
    aerosol_density: float
    cloud_seed: int

    def transmittance(self, distance_m: float) -> float:
        """Return the fraction of contrast surviving over a distance.

        Uses Koschmieder's law with a 2 % contrast threshold.

        Args:
            distance_m: Path length through the haze in metres.

        Returns:
            The transmittance in ``[0, 1]``.
        """
        return math.exp(-3.912 * distance_m / self.visibility_m)


@dataclass(frozen=True)
class SeaState:
    """Wave field parameters.

    Attributes:
        beaufort: Beaufort number of the sea state.
        wind_speed_ms: Wind speed driving the wave spectrum.
        wind_direction_deg: Compass bearing the waves travel towards.
        choppiness: Horizontal displacement factor of wave crests.
        wave_alignment: How strongly the waves align with the wind (0-1).
        foam_coverage: Amount of whitecaps.
        water_color: Linear-RGB base colour of the water body.
        ocean_seed: Seed of the wave spectrum.
        time_s: Simulation time of the wave field.
    """

    beaufort: int
    wind_speed_ms: float
    wind_direction_deg: float
    choppiness: float
    wave_alignment: float
    foam_coverage: float
    water_color: tuple[float, float, float]
    ocean_seed: int
    time_s: float

    @property
    def significant_wave_height_m(self) -> float:
        """Significant wave height interpolated from the WMO Beaufort table."""
        speeds = [0.5 * (low + high) for low, high in BEAUFORT_WIND_MS]
        heights = BEAUFORT_WAVE_HEIGHT_M
        if self.wind_speed_ms <= speeds[0]:
            return heights[0]
        for (s0, h0), (s1, h1) in itertools.pairwise(zip(speeds, heights, strict=True)):
            if self.wind_speed_ms <= s1:
                return h0 + (h1 - h0) * (self.wind_speed_ms - s0) / (s1 - s0)
        return heights[-1]

    @property
    def peak_wavelength_m(self) -> float:
        """Peak wavelength of a fully developed Pierson-Moskowitz sea."""
        return max(0.834 * self.wind_speed_ms**2, 0.5)


@dataclass(frozen=True)
class Land:
    """An optional distant coastline silhouette.

    Attributes:
        present: Whether land appears in this scenario.
        bearing_deg: Compass bearing of the coastline's centre.
        width_deg: Angular width of the coastline along the horizon.
        height_m: Silhouette height above mean sea level in metres.
        seed: Seed for the terrain shape.
    """

    present: bool
    bearing_deg: float = 0.0
    width_deg: float = 0.0
    height_m: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class PlacedObject:
    """An object placed on the sea surface.

    Attributes:
        category: Object class name (see :data:`sea_data.config.CLASS_NAMES`).
        length_m: Overall length (height for buoys) in metres.
        distance_m: Horizontal range from the camera.
        bearing_deg: Compass bearing from the camera.
        heading_deg: Compass heading the object's bow points to.
        variant_seed: Seed for shape and colour variations.
    """

    category: str
    length_m: float
    distance_m: float
    bearing_deg: float
    heading_deg: float
    variant_seed: int

    @property
    def height_m(self) -> float:
        """Approximate height of the object above the waterline."""
        return self.length_m * OBJECT_HEIGHT_RATIO[self.category]


@dataclass(frozen=True)
class Scenario:
    """Every random choice needed to render one image.

    Attributes:
        index: Image index within the dataset.
        seed: Dataset seed the scenario was drawn from.
        platform: Name of the camera platform preset.
        hfov_deg: Horizontal field of view.
        camera: Camera pose.
        sun: Sun position.
        atmosphere: Sky and visibility conditions.
        sea: Wave field parameters.
        objects: Objects placed in the scene.
        land: Optional distant coastline silhouette.
        exposure_offset_ev: Random exposure deviation from the auto-exposure.
    """

    index: int
    seed: int
    platform: str
    hfov_deg: float
    camera: CameraPose
    sun: Sun
    atmosphere: Atmosphere
    sea: SeaState
    objects: tuple[PlacedObject, ...]
    land: Land
    exposure_offset_ev: float = 0.0

    @property
    def stem(self) -> str:
        """File name stem used for every output of this scenario."""
        return f"{self.index:06d}"

    def to_dict(self) -> dict[str, Any]:
        """Return the scenario as JSON-serialisable nested mappings.

        Returns:
            The scenario as plain dictionaries, lists and scalars.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scenario:
        """Rebuild a scenario from :meth:`to_dict` output.

        Args:
            data: Mapping produced by :meth:`to_dict` (possibly via JSON).

        Returns:
            The reconstructed scenario.
        """
        sea = dict(data["sea"])
        sea["water_color"] = tuple(sea["water_color"])
        return cls(
            index=data["index"],
            seed=data["seed"],
            platform=data["platform"],
            hfov_deg=data["hfov_deg"],
            camera=CameraPose(**data["camera"]),
            sun=Sun(**data["sun"]),
            atmosphere=Atmosphere(**data["atmosphere"]),
            sea=SeaState(**sea),
            objects=tuple(PlacedObject(**o) for o in data["objects"]),
            land=Land(**data["land"]),
            exposure_offset_ev=data.get("exposure_offset_ev", 0.0),
        )


def _choose[T](
    rng: random.Random, options: Mapping[str, T], weight: Callable[[T], float]
) -> str:
    names = list(options)
    weights = [weight(options[name]) for name in names]
    return rng.choices(names, weights=weights)[0]


def _uniform(rng: random.Random, interval: Sequence[float]) -> float:
    return rng.uniform(interval[0], interval[1])


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _sample_sea(rng: random.Random, config: SimulatorConfig) -> SeaState:
    beaufort = rng.choices(range(9), weights=config.beaufort_weights)[0]
    wind = rng.uniform(*BEAUFORT_WIND_MS[beaufort])
    base = rng.choice(WATER_COLORS)
    tint = rng.uniform(0.75, 1.3)
    return SeaState(
        beaufort=beaufort,
        wind_speed_ms=max(wind, 0.3),
        wind_direction_deg=rng.uniform(0.0, 360.0),
        choppiness=min(0.4 + 0.12 * beaufort + rng.uniform(-0.1, 0.2), 1.6),
        wave_alignment=rng.uniform(0.0, 0.8),
        foam_coverage=max(0.0, 0.08 * (beaufort - 3)) * rng.uniform(0.7, 1.3),
        water_color=(base[0] * tint, base[1] * tint, base[2] * tint),
        ocean_seed=rng.randrange(1, 2**16),
        time_s=rng.uniform(0.0, 1000.0),
    )


def _sample_land(
    rng: random.Random, config: SimulatorConfig, camera: CameraPose, hfov_deg: float
) -> Land:
    if rng.random() >= config.land.probability:
        return Land(present=False)
    max_width = max(10.0, 0.75 * hfov_deg)
    return Land(
        present=True,
        bearing_deg=(camera.yaw_deg + rng.uniform(-0.4, 0.4) * hfov_deg) % 360.0,
        width_deg=min(_uniform(rng, config.land.width_deg), max_width),
        height_m=_uniform(rng, config.land.height_m),
        seed=rng.randrange(2**16),
    )


def _overlaps(candidate: PlacedObject, placed: Sequence[PlacedObject]) -> bool:
    cb = math.radians(candidate.bearing_deg)
    cx, cy = candidate.distance_m * math.sin(cb), candidate.distance_m * math.cos(cb)
    for other in placed:
        ob = math.radians(other.bearing_deg)
        ox, oy = other.distance_m * math.sin(ob), other.distance_m * math.cos(ob)
        clearance = 0.55 * (candidate.length_m + other.length_m) + 2.0
        if math.hypot(cx - ox, cy - oy) < clearance:
            return True
    return False


def _sample_objects(
    rng: random.Random, config: SimulatorConfig, camera: CameraPose, hfov_deg: float
) -> tuple[PlacedObject, ...]:
    count = rng.randint(*config.objects_per_image)
    classes = config.object_classes
    camera_range = horizon_distance(camera.height_m)
    placed: list[PlacedObject] = []
    for _ in range(count):
        for _attempt in range(_SEPARATION_ATTEMPTS):
            category = _choose(rng, classes, lambda c: c.weight)
            length = _uniform(rng, classes[category].length_m)
            height = length * OBJECT_HEIGHT_RATIO[category]
            near = max(config.min_object_distance_m, 1.2 * length)
            far = config.max_range_factor * (camera_range + horizon_distance(height))
            candidate = PlacedObject(
                category=category,
                length_m=length,
                distance_m=_log_uniform(rng, near, max(far, 1.5 * near)),
                bearing_deg=(
                    camera.yaw_deg + rng.uniform(-0.5, 0.5) * (hfov_deg + 10.0)
                )
                % 360.0,
                heading_deg=rng.uniform(0.0, 360.0),
                variant_seed=rng.randrange(2**31),
            )
            if not _overlaps(candidate, placed):
                placed.append(candidate)
                break
    return tuple(placed)


def sample_scenario(config: SimulatorConfig, seed: int, index: int) -> Scenario:
    """Draw one scenario deterministically from ``(seed, index)``.

    Args:
        config: Randomisation ranges.
        seed: Dataset seed.
        index: Image index; different indices give independent scenarios.

    Returns:
        The sampled scenario.
    """
    rng = random.Random(f"sea-data:{seed}:{index}")
    weather_name = _choose(rng, config.weather, lambda w: w.weight)
    weather = config.weather[weather_name]
    platform_name = _choose(rng, config.platforms, lambda p: p.weight)
    platform = config.platforms[platform_name]
    hfov = _uniform(rng, platform.hfov_deg)
    camera = CameraPose(
        height_m=_log_uniform(rng, *platform.height_m),
        yaw_deg=rng.uniform(0.0, 360.0),
        pitch_deg=_uniform(rng, platform.pitch_deg),
        roll_deg=_uniform(rng, platform.roll_deg),
    )
    atmosphere = Atmosphere(
        weather=weather_name,
        visibility_m=_log_uniform(rng, *weather.visibility_m),
        cloud_cover=_uniform(rng, weather.cloud_cover),
        aerosol_density=_uniform(rng, weather.aerosol_density),
        cloud_seed=rng.randrange(2**16),
    )
    sun = Sun(
        elevation_deg=_uniform(rng, config.sun_elevation_deg),
        azimuth_deg=rng.uniform(0.0, 360.0),
    )
    sea = _sample_sea(rng, config)
    objects = _sample_objects(rng, config, camera, hfov)
    land = _sample_land(rng, config, camera, hfov)
    return Scenario(
        index=index,
        seed=seed,
        platform=platform_name,
        hfov_deg=hfov,
        camera=camera,
        sun=sun,
        atmosphere=atmosphere,
        sea=sea,
        objects=objects,
        land=land,
        exposure_offset_ev=rng.gauss(0.0, config.render.exposure_jitter_ev),
    )


def sample_scenarios(
    config: SimulatorConfig, seed: int, count: int, start: int = 0
) -> Iterator[Scenario]:
    """Lazily draw a sequence of scenarios.

    Args:
        config: Randomisation ranges.
        seed: Dataset seed.
        count: Number of scenarios.
        start: Index of the first scenario.

    Yields:
        Scenarios with indices ``start`` to ``start + count - 1``.
    """
    for index in range(start, start + count):
        yield sample_scenario(config, seed, index)
