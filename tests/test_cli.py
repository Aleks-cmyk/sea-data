import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from sea_data import cli
from sea_data.config import SimulatorConfig
from sea_data.scenario import sample_scenarios


def test_sample_prints_scenarios(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["sample", "-n", "2", "--seed", "4", "--start", "5"]) == 0
    scenarios = json.loads(capsys.readouterr().out)
    assert [s["index"] for s in scenarios] == [5, 6]
    assert all(s["seed"] == 4 for s in scenarios)


def test_sample_reports_config_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("nonsense = 1\n", encoding="utf-8")
    assert cli.main(["sample", "-c", str(path)]) == 2
    assert "unknown configuration key" in capsys.readouterr().err


def test_index_on_empty_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["index", str(tmp_path)]) == 0
    assert "0 images" in capsys.readouterr().out
    assert (tmp_path / "annotations_coco.json").exists()


def test_generate_without_blender_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["generate", "-o", str(tmp_path), "-n", "1", "--blender", "no-such-blender-x"]
    )
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_generate_writes_jobs_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[Path] = []

    def fake_run(blender: str, jobs: Sequence[Path]) -> int:
        launched.extend(jobs)
        return 0

    monkeypatch.setattr(cli, "find_blender", lambda explicit=None: "blender")
    monkeypatch.setattr(cli, "run_workers", fake_run)
    code = cli.main(
        [
            "generate",
            "-o",
            str(tmp_path),
            "-n",
            "3",
            "-j",
            "2",
            "--width",
            "320",
            "--engine",
            "BLENDER_EEVEE",
        ]
    )
    assert code == 0
    assert [p.name for p in launched] == ["job_00.json", "job_01.json"]
    job = json.loads(launched[0].read_text())
    assert job["render"]["width"] == 320
    assert job["render"]["engine"] == "BLENDER_EEVEE"
    assert [s["index"] for s in job["scenarios"]] == [0, 2]
    assert json.loads((tmp_path / "config.json").read_text())["render"]["width"] == 320
    assert (tmp_path / "horizon.csv").exists()


def test_write_jobs_skips_empty_chunks(tmp_path: Path) -> None:
    config = SimulatorConfig()
    scenarios = list(sample_scenarios(config, 0, 2))
    jobs = cli.write_jobs(config, scenarios, tmp_path, workers=4, overwrite=True)
    assert len(jobs) == 2
    assert json.loads(jobs[1].read_text())["overwrite"] is True


def test_blender_environment_drops_virtualenv() -> None:
    env = cli.blender_environment(
        {
            "VIRTUAL_ENV": "/proj/.venv",
            "PYTHONPATH": "/x",
            "PATH": "/proj/.venv/bin:/usr/bin",
            "HOME": "/home/me",
        }
    )
    assert env == {"PATH": "/usr/bin", "HOME": "/home/me"}


def test_find_blender_honours_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cli.BLENDER_ENV, "sh")
    assert Path(cli.find_blender()).name == "sh"
    with pytest.raises(FileNotFoundError):
        cli.find_blender("no-such-blender-x")
