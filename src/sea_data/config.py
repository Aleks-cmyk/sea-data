"""Simulator configuration: randomisation ranges, presets and render settings.

Every field has a sensible default. A TOML file can override any subset of
them, for example::

    [render]
    width = 1920
    height = 1080
    samples = 128

    [weather.fog]
    weight = 0.5

    [object_classes.buoy]
    length_m = [2.0, 6.0]

Presets (weather, platforms, object classes) are merged by name. A new preset
must define all of its fields; a preset is disabled by setting its weight to 0.
"""

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

type Interval = tuple[float, float]

CLASS_NAMES: tuple[str, ...] = (
    "cargo_ship",
    "fishing_boat",
    "sailboat",
    "motorboat",
    "buoy",
)
"""Object categories the procedural builder can generate, in label order."""

SEMANTIC_SKY = 0
"""Semantic mask value of sky pixels."""
SEMANTIC_SEA = 1
"""Semantic mask value of sea pixels."""
SEMANTIC_OBJECT_OFFSET = 2
"""Semantic mask value of the first object class; class ``i`` uses ``2 + i``."""

ENGINES = ("CYCLES", "BLENDER_EEVEE")


class ConfigError(ValueError):
    """Raised when a configuration file contains invalid values."""


@dataclass(frozen=True)
class WeatherPreset:
    """Randomisation ranges for one kind of weather.

    Attributes:
        weight: Relative sampling probability.
        visibility_m: Meteorological visibility range in metres.
        cloud_cover: Fraction of the sky covered by clouds.
        aerosol_density: Aerosol density of the physical sky model.
    """

    weight: float
    visibility_m: Interval
    cloud_cover: Interval
    aerosol_density: Interval


@dataclass(frozen=True)
class PlatformPreset:
    """Randomisation ranges for one kind of camera platform.

    Attributes:
        weight: Relative sampling probability.
        height_m: Camera height above mean sea level in metres.
        pitch_deg: Camera pitch in degrees, positive upwards.
        roll_deg: Camera roll in degrees.
        hfov_deg: Horizontal field of view in degrees.
    """

    weight: float
    height_m: Interval
    pitch_deg: Interval
    roll_deg: Interval
    hfov_deg: Interval


@dataclass(frozen=True)
class ObjectClassConfig:
    """Randomisation ranges for one object category.

    Attributes:
        weight: Relative sampling probability.
        length_m: Overall length (height for buoys) in metres.
    """

    weight: float
    length_m: Interval


@dataclass(frozen=True)
class RenderConfig:
    """Render settings.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        engine: ``"CYCLES"`` (path traced) or ``"BLENDER_EEVEE"`` (raster).
        samples: Samples per pixel.
        device: ``"CPU"`` or ``"GPU"`` (Cycles only).
        denoise: Whether to denoise Cycles renders.
        image_format: ``"PNG"`` or ``"JPEG"``.
        exposure_jitter_ev: Standard deviation of the random exposure offset.
    """

    width: int = 1280
    height: int = 720
    engine: str = "CYCLES"
    samples: int = 48
    device: str = "CPU"
    denoise: bool = True
    image_format: str = "PNG"
    exposure_jitter_ev: float = 0.3


@dataclass(frozen=True)
class AnnotationConfig:
    """Rules for turning rendered objects into training labels.

    Attributes:
        min_visible_pixels: Objects with fewer visible pixels are not labelled.
        min_transmittance: Objects whose atmospheric transmittance is lower
            (i.e. hidden by fog or haze) are not labelled.
        min_horizon_transmittance: The horizon is marked invisible when the
            atmospheric transmittance over its distance is lower than this,
            i.e. when fog or haze would blend it into the sky so much that a
            human could not point to it.
    """

    min_visible_pixels: int = 12
    min_transmittance: float = 0.03
    min_horizon_transmittance: float = 0.05


@dataclass(frozen=True)
class LandConfig:
    """Randomisation ranges for an optional distant coastline.

    A low silhouette is placed along part of the horizon, never across its
    full width, so the sea horizon stays visible on at least one side.

    Attributes:
        probability: Chance that land appears in a given image.
        height_m: Silhouette height above mean sea level in metres.
        width_deg: Angular width of the coastline along the horizon.
    """

    probability: float = 0.2
    height_m: Interval = (5.0, 80.0)
    width_deg: Interval = (15.0, 70.0)


def _default_weather() -> dict[str, WeatherPreset]:
    return {
        "clear": WeatherPreset(0.40, (20_000.0, 60_000.0), (0.0, 0.35), (0.5, 2.0)),
        "hazy": WeatherPreset(0.25, (4_000.0, 15_000.0), (0.0, 0.5), (2.0, 6.0)),
        "overcast": WeatherPreset(0.20, (8_000.0, 30_000.0), (0.75, 1.0), (1.0, 4.0)),
        "fog": WeatherPreset(0.15, (300.0, 2_500.0), (0.85, 1.0), (4.0, 10.0)),
    }


def _default_platforms() -> dict[str, PlatformPreset]:
    return {
        "small_boat": PlatformPreset(
            0.25, (1.5, 3.5), (-6.0, 4.0), (-10.0, 10.0), (50.0, 95.0)
        ),
        "ship_bridge": PlatformPreset(
            0.35, (8.0, 30.0), (-5.0, 2.0), (-4.0, 4.0), (35.0, 80.0)
        ),
        "shore_tower": PlatformPreset(
            0.15, (10.0, 60.0), (-6.0, 1.0), (-1.0, 1.0), (20.0, 70.0)
        ),
        "buoy": PlatformPreset(
            0.10, (1.0, 2.0), (-8.0, 8.0), (-15.0, 15.0), (60.0, 100.0)
        ),
        "drone": PlatformPreset(
            0.15, (30.0, 120.0), (-25.0, -5.0), (-3.0, 3.0), (60.0, 90.0)
        ),
    }


def _default_object_classes() -> dict[str, ObjectClassConfig]:
    return {
        "cargo_ship": ObjectClassConfig(0.20, (80.0, 300.0)),
        "fishing_boat": ObjectClassConfig(0.20, (10.0, 35.0)),
        "sailboat": ObjectClassConfig(0.20, (7.0, 22.0)),
        "motorboat": ObjectClassConfig(0.25, (4.0, 15.0)),
        "buoy": ObjectClassConfig(0.15, (1.5, 5.0)),
    }


@dataclass(frozen=True)
class SimulatorConfig:
    """Complete simulator configuration.

    Attributes:
        render: Render settings.
        annotation: Labelling rules.
        weather: Weather presets by name.
        platforms: Camera platform presets by name.
        object_classes: Object category settings by class name.
        land: Randomisation ranges for the optional distant coastline.
        beaufort_weights: Sampling weights of Beaufort sea states 0 to 8.
        sun_elevation_deg: Sun elevation range; negative values give twilight.
        objects_per_image: Inclusive range of objects placed per image.
        min_object_distance_m: Minimum horizontal range of an object.
        max_range_factor: Objects are placed up to this factor times the
            combined geometric horizon range of camera and object, so that
            some of them are partially hidden behind the horizon ("hull-down").
    """

    render: RenderConfig = field(default_factory=RenderConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    weather: dict[str, WeatherPreset] = field(default_factory=_default_weather)
    platforms: dict[str, PlatformPreset] = field(default_factory=_default_platforms)
    object_classes: dict[str, ObjectClassConfig] = field(
        default_factory=_default_object_classes
    )
    land: LandConfig = field(default_factory=LandConfig)
    beaufort_weights: tuple[float, ...] = (
        0.04,
        0.10,
        0.16,
        0.18,
        0.18,
        0.14,
        0.10,
        0.06,
        0.04,
    )
    sun_elevation_deg: Interval = (-3.0, 65.0)
    objects_per_image: tuple[int, int] = (0, 6)
    min_object_distance_m: float = 25.0
    max_range_factor: float = 1.1

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as JSON-serialisable nested mappings.

        Returns:
            The configuration as plain dictionaries, lists and scalars.
        """
        return asdict(self)

    def validate(self) -> None:
        """Check cross-field constraints.

        Raises:
            ConfigError: If a value is out of range or unsupported.
        """
        unknown = set(self.object_classes) - set(CLASS_NAMES)
        if unknown:
            raise ConfigError(f"unsupported object classes: {sorted(unknown)}")
        if self.render.engine not in ENGINES:
            raise ConfigError(f"render.engine must be one of {ENGINES}")
        if self.render.image_format not in ("PNG", "JPEG"):
            raise ConfigError("render.image_format must be PNG or JPEG")
        if len(self.beaufort_weights) != 9:
            raise ConfigError("beaufort_weights needs 9 entries (Beaufort 0-8)")
        low, high = self.objects_per_image
        if not 0 <= low <= high <= 254:
            raise ConfigError("objects_per_image must satisfy 0 <= low <= high <= 254")
        if not 0.0 <= self.land.probability <= 1.0:
            raise ConfigError("land.probability must be between 0 and 1")
        for name, presets in (
            ("weather", self.weather),
            ("platforms", self.platforms),
            ("object_classes", self.object_classes),
        ):
            if not any(p.weight > 0 for p in presets.values()):
                raise ConfigError(f"{name} needs at least one preset with weight > 0")


def _coerce(current: object, value: Any, key: str) -> Any:
    if is_dataclass(current) and not isinstance(current, type):
        return _merge(current, value, key)
    if isinstance(current, dict):
        return _merge_presets(current, value, key)
    if isinstance(current, tuple):
        if not current:
            return tuple(value)
        element_type = type(current[0])
        return tuple(element_type(v) for v in value)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return value
    return type(current)(value)


def _merge[T](base: T, data: Mapping[str, Any], path: str = "") -> T:
    try:
        items = data.items()
    except AttributeError as exc:
        raise ConfigError(f"{path or 'config'} must be a table") from exc
    updates: dict[str, Any] = {}
    for key, value in items:
        dotted = f"{path}.{key}" if path else key
        try:
            current = getattr(base, key)
        except AttributeError as exc:
            raise ConfigError(f"unknown configuration key: {dotted}") from exc
        try:
            updates[key] = _coerce(current, value, dotted)
        except ConfigError:
            raise
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid value for {dotted}: {value!r}") from exc
    return replace(base, **updates)  # type: ignore[type-var]


def _merge_presets(
    current: dict[str, Any], data: Mapping[str, Any], path: str
) -> dict[str, Any]:
    preset_type = type(next(iter(current.values())))
    merged = dict(current)
    try:
        items = data.items()
    except AttributeError as exc:
        raise ConfigError(f"{path} must be a table") from exc
    for name, values in items:
        dotted = f"{path}.{name}"
        try:
            merged[name] = _merge(current[name], values, dotted)
        except KeyError:
            names = {f.name for f in fields(preset_type)}
            missing = names - set(values)
            if missing:
                raise ConfigError(
                    f"new preset {dotted} is missing {sorted(missing)}"
                ) from None
            converted = {
                key: tuple(v) if isinstance(v, list) else v for key, v in values.items()
            }
            try:
                merged[name] = preset_type(**converted)
            except TypeError as exc:
                raise ConfigError(f"invalid preset {dotted}: {exc}") from exc
    return merged


def config_from_mapping(data: Mapping[str, Any]) -> SimulatorConfig:
    """Build a configuration by overriding the defaults with a mapping.

    Args:
        data: Nested mapping, e.g. parsed from TOML or JSON.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the mapping contains unknown keys or invalid values.
    """
    config = _merge(SimulatorConfig(), data)
    config.validate()
    return config


def load_config(path: Path | None = None) -> SimulatorConfig:
    """Load a configuration file, or return the defaults.

    Args:
        path: Path to a TOML file. ``None`` returns the default configuration.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is not valid TOML or has invalid values.
    """
    if path is None:
        return SimulatorConfig()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return config_from_mapping(data)
