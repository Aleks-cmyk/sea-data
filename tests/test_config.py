from pathlib import Path

import pytest

from sea_data.config import (
    CLASS_NAMES,
    ConfigError,
    SimulatorConfig,
    WeatherPreset,
    config_from_mapping,
    load_config,
)


def test_defaults_are_valid() -> None:
    config = load_config()
    config.validate()
    assert set(config.object_classes) == set(CLASS_NAMES)
    assert config.to_dict()["render"]["width"] == 1280


def test_toml_overrides_merge_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.toml"
    path.write_text(
        """
objects_per_image = [1, 3]

[render]
width = 640
samples = 8

[weather.fog]
weight = 0.9

[weather.storm]
weight = 0.1
visibility_m = [1000.0, 4000.0]
cloud_cover = [0.9, 1.0]
aerosol_density = [3.0, 6.0]

[object_classes.buoy]
length_m = [2, 6]
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.render.width == 640
    assert config.render.height == 720
    assert config.render.samples == 8
    assert config.objects_per_image == (1, 3)
    assert config.weather["fog"].weight == 0.9
    assert config.weather["fog"].visibility_m == (300.0, 2_500.0)
    assert config.weather["storm"] == WeatherPreset(
        0.1, (1000.0, 4000.0), (0.9, 1.0), (3.0, 6.0)
    )
    assert config.object_classes["buoy"].length_m == (2.0, 6.0)
    assert isinstance(config.object_classes["buoy"].length_m[0], float)


@pytest.mark.parametrize(
    "data",
    [
        {"render": {"colour": 1}},
        {"unknown": 1},
        {"render": {"engine": "LUXCORE"}},
        {"render": {"width": "wide"}},
        {"render": 5},
        {"weather": {"storm": {"weight": 1.0}}},
        {"object_classes": {"submarine": {"weight": 1.0, "length_m": [1, 2]}}},
        {"objects_per_image": [4, 2]},
        {"beaufort_weights": [1.0, 2.0]},
        {"platforms": {name: {"weight": 0.0} for name in SimulatorConfig().platforms}},
    ],
)
def test_invalid_configurations_are_rejected(data: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        config_from_mapping(data)


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("[render\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
