"""Script executed by Blender to render a job file.

Usage::

    blender --background --factory-startup --python blender_worker.py -- --job job.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sea_data.render import run_job

if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    sys.exit(run_job(argv))
