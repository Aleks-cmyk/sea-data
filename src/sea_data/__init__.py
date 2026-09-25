"""Synthetic maritime data for object and horizon detection, rendered in Blender."""

import sys

from sea_data.cli import main as _cli_main


def main() -> None:
    """Console-script entry point of ``sea-data``."""
    sys.exit(_cli_main())
