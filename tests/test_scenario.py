import json
import math

import pytest

from sea_data.config import SimulatorConfig, config_from_mapping
from sea_data.geometry import horizon_distance
from sea_data.scenario import (
    BEAUFORT_WAVE_HEIGHT_M,
    BEAUFORT_WIND_MS,
    Atmosphere,
    Scenario,
    SeaState,
    sample_scenario,
    sample_scenarios,
)


def _sea(wind: float) -> SeaState:
    return SeaState(3, wind, 0.0, 1.0, 0.5, 0.0, (0.0, 0.0, 0.0), 1, 0.0)


def test_sampling_is_deterministic_per_seed_and_index() -> None:
    config = SimulatorConfig()
    assert sample_scenario(config, 7, 3) == sample_scenario(config, 7, 3)
    assert sample_scenario(config, 7, 3) != sample_scenario(config, 7, 4)
    assert sample_scenario(config, 7, 3) != sample_scenario(config, 8, 3)


def test_sample_scenarios_indices() -> None:
    scenarios = list(sample_scenarios(SimulatorConfig(), 1, 3, start=10))
    assert [s.index for s in scenarios] == [10, 11, 12]
    assert scenarios[0].stem == "000010"


def test_json_round_trip() -> None:
    config = config_from_mapping({"objects_per_image": [3, 3]})
    scenario = sample_scenario(config, 0, 0)
    restored = Scenario.from_dict(json.loads(json.dumps(scenario.to_dict())))
    assert restored == scenario


def test_samples_respect_configured_ranges() -> None:
    config = config_from_mapping({"objects_per_image": [2, 5]})
    for index in range(50):
        scenario = sample_scenario(config, 3, index)
        platform = config.platforms[scenario.platform]
        assert platform.height_m[0] <= scenario.camera.height_m <= platform.height_m[1]
        assert platform.hfov_deg[0] <= scenario.hfov_deg <= platform.hfov_deg[1]
        weather = config.weather[scenario.atmosphere.weather]
        assert (
            weather.visibility_m[0]
            <= scenario.atmosphere.visibility_m
            <= weather.visibility_m[1]
        )
        assert 0 <= scenario.sea.beaufort <= 8
        low, high = BEAUFORT_WIND_MS[scenario.sea.beaufort]
        assert max(low, 0.3) <= scenario.sea.wind_speed_ms <= max(high, 0.3)
        assert len(scenario.objects) <= 5
        for obj in scenario.objects:
            rel = (obj.bearing_deg - scenario.camera.yaw_deg + 180.0) % 360.0 - 180.0
            assert abs(rel) <= 0.5 * (scenario.hfov_deg + 10.0) + 1e-9
            near = max(config.min_object_distance_m, 1.2 * obj.length_m)
            far = config.max_range_factor * (
                horizon_distance(scenario.camera.height_m)
                + horizon_distance(obj.height_m)
            )
            assert near <= obj.distance_m <= max(far, 1.5 * near)


def test_objects_do_not_overlap() -> None:
    config = config_from_mapping({"objects_per_image": [6, 6]})
    for index in range(20):
        objects = sample_scenario(config, 5, index).objects
        for i, a in enumerate(objects):
            for b in objects[i + 1 :]:
                ax = a.distance_m * math.sin(math.radians(a.bearing_deg))
                ay = a.distance_m * math.cos(math.radians(a.bearing_deg))
                bx = b.distance_m * math.sin(math.radians(b.bearing_deg))
                by = b.distance_m * math.cos(math.radians(b.bearing_deg))
                assert math.hypot(ax - bx, ay - by) >= 0.55 * (a.length_m + b.length_m)


def test_disabled_presets_are_never_sampled() -> None:
    config = config_from_mapping(
        {"weather": {name: {"weight": 0.0} for name in ("clear", "hazy", "overcast")}}
    )
    weathers = {sample_scenario(config, 0, i).atmosphere.weather for i in range(30)}
    assert weathers == {"fog"}


def test_significant_wave_height_follows_beaufort_table() -> None:
    for (low, high), height in zip(
        BEAUFORT_WIND_MS, BEAUFORT_WAVE_HEIGHT_M, strict=True
    ):
        assert _sea(0.5 * (low + high)).significant_wave_height_m == pytest.approx(
            height
        )
    heights = [_sea(w).significant_wave_height_m for w in range(30)]
    assert heights == sorted(heights)
    assert _sea(50.0).significant_wave_height_m == BEAUFORT_WAVE_HEIGHT_M[-1]


def test_transmittance_follows_koschmieder() -> None:
    atmosphere = Atmosphere("fog", 1000.0, 1.0, 5.0, 0)
    assert atmosphere.transmittance(0.0) == 1.0
    assert atmosphere.transmittance(1000.0) == pytest.approx(0.02, abs=1e-3)
