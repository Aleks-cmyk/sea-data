"""Command-line interface of the maritime data simulator.

Subcommands:
    * ``generate``: sample scenarios, render them in Blender, build the index.
    * ``sample``: print sampled scenarios as JSON without rendering.
    * ``index``: rebuild COCO/YOLO/horizon files from existing metadata.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from sea_data.annotations import build_index
from sea_data.config import ConfigError, SimulatorConfig, load_config
from sea_data.scenario import Scenario, sample_scenarios

WORKER_SCRIPT = Path(__file__).with_name("blender_worker.py")
JOB_DIR = "jobs"
BLENDER_ENV = "SEA_DATA_BLENDER"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sea-data",
        description="Synthetic maritime images with object and horizon ground truth.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-n", "--count", type=int, default=10, help="number of images")
        p.add_argument("--seed", type=int, default=0, help="dataset seed")
        p.add_argument("--start", type=int, default=0, help="index of the first image")
        p.add_argument("-c", "--config", type=Path, help="TOML configuration file")

    gen = sub.add_parser("generate", help="render a dataset with Blender")
    add_common(gen)
    gen.add_argument("-o", "--output", type=Path, required=True, help="dataset folder")
    gen.add_argument("--width", type=int, help="image width override")
    gen.add_argument("--height", type=int, help="image height override")
    gen.add_argument("--samples", type=int, help="render samples override")
    gen.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), help="renderer")
    gen.add_argument("--device", choices=("CPU", "GPU"), help="Cycles device")
    gen.add_argument("-j", "--workers", type=int, default=1, help="Blender processes")
    gen.add_argument("--blender", help=f"Blender executable (default: ${BLENDER_ENV})")
    gen.add_argument("--overwrite", action="store_true", help="re-render existing")

    sample = sub.add_parser("sample", help="print sampled scenarios as JSON")
    add_common(sample)

    index = sub.add_parser("index", help="rebuild annotation files of a dataset")
    index.add_argument("output", type=Path, help="dataset folder")
    return parser


def _render_overrides(
    config: SimulatorConfig, args: argparse.Namespace
) -> SimulatorConfig:
    overrides = {
        name: value
        for name in ("width", "height", "samples", "engine", "device")
        if (value := getattr(args, name)) is not None
    }
    return replace(config, render=replace(config.render, **overrides))


def find_blender(explicit: str | None = None) -> str:
    """Locate the Blender executable.

    Args:
        explicit: Path given on the command line, if any.

    Returns:
        The executable to run.

    Raises:
        FileNotFoundError: If Blender cannot be found.
    """
    candidate = explicit or os.environ.get(BLENDER_ENV) or "blender"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"Blender executable {candidate!r} not found; use --blender or ${BLENDER_ENV}"
        )
    return resolved


def write_jobs(
    config: SimulatorConfig,
    scenarios: Sequence[Scenario],
    output: Path,
    workers: int,
    overwrite: bool = False,
) -> list[Path]:
    """Split scenarios into job files, one per Blender worker.

    Args:
        config: Simulator configuration (render and annotation settings).
        scenarios: Scenarios to render.
        output: Dataset folder.
        workers: Number of Blender processes.
        overwrite: Whether workers re-render images that already exist.

    Returns:
        Paths of the written job files (empty chunks are skipped).
    """
    job_dir = output / JOB_DIR
    job_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for worker in range(max(workers, 1)):
        chunk = scenarios[worker :: max(workers, 1)]
        if not chunk:
            continue
        job = {
            "output_dir": str(output.resolve()),
            "render": asdict(config.render),
            "annotation": asdict(config.annotation),
            "overwrite": overwrite,
            "scenarios": [s.to_dict() for s in chunk],
        }
        path = job_dir / f"job_{worker:02d}.json"
        path.write_text(json.dumps(job), encoding="utf-8")
        paths.append(path)
    return paths


def blender_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment in which Blender uses its own Python setup.

    When launched from an activated virtual environment (e.g. ``uv run``),
    Blender's embedded interpreter would pick up the virtual environment and
    lose its bundled packages such as ``numpy``.

    Args:
        environ: Base environment; defaults to ``os.environ``.

    Returns:
        A copy without virtual-environment and Python path variables.
    """
    env = dict(os.environ if environ is None else environ)
    venv = env.pop("VIRTUAL_ENV", None)
    for name in ("PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX"):
        env.pop(name, None)
    if venv:
        venv_bin = str(Path(venv) / "bin")
        env["PATH"] = os.pathsep.join(
            entry
            for entry in env.get("PATH", "").split(os.pathsep)
            if entry != venv_bin
        )
    return env


def run_workers(blender: str, jobs: Sequence[Path]) -> int:
    """Run one Blender process per job file and wait for all of them.

    Args:
        blender: Blender executable.
        jobs: Job files from :func:`write_jobs`.

    Returns:
        Number of workers that exited with an error.
    """
    processes = [
        subprocess.Popen(
            [
                blender,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(WORKER_SCRIPT),
                "--",
                "--job",
                str(job),
            ],
            env=blender_environment(),
        )
        for job in jobs
    ]
    return sum(1 for process in processes if process.wait() != 0)


def _generate(args: argparse.Namespace, config: SimulatorConfig) -> int:
    config = _render_overrides(config, args)
    config.validate()
    blender = find_blender(args.blender)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=1), encoding="utf-8"
    )
    scenarios = list(sample_scenarios(config, args.seed, args.count, args.start))
    jobs = write_jobs(config, scenarios, output, args.workers, args.overwrite)
    failed = run_workers(blender, jobs)
    summary = build_index(output)
    print(
        f"{summary.images} images, {summary.objects} labelled objects, "
        f"{summary.horizons} visible horizons -> {output}"
    )
    if failed:
        print(f"{failed} Blender worker(s) reported errors", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``sea-data`` command line.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = _parser().parse_args(argv)
    try:
        if args.command == "index":
            summary = build_index(args.output)
            print(
                f"{summary.images} images, {summary.objects} labelled objects, "
                f"{summary.horizons} visible horizons"
            )
            return 0
        config = load_config(args.config)
        if args.command == "sample":
            scenarios = sample_scenarios(config, args.seed, args.count, args.start)
            print(json.dumps([s.to_dict() for s in scenarios], indent=1))
            return 0
        return _generate(args, config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"sea-data: error: {exc}", file=sys.stderr)
        return 2
